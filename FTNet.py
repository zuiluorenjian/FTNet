#!/usr/bin/env python3
import os
import yaml
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from networks.clip.clip import load as clip_load

from sklearn.metrics import accuracy_score, roc_auc_score


# ==========================================
# Utils
# ==========================================

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


# ==========================================
# Dataset
# ==========================================

class CacheDataset(Dataset):
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.preprocess(self.image_paths[idx])
        label = self.labels[idx]
        return image, label


# ==========================================
# FTNet
# ==========================================

class FTNet:

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model, self.preprocess = clip_load(
            self.config["clip_model"],
            device=self.device
        )
        self.clip_model.eval()

        # =====================================================
        # Feature dimension and dtype inference (unchanged logic)
        # =====================================================

        extract_layer = config.get('clip_layer', None)

        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)

            if extract_layer is not None:
                try:
                    if hasattr(self.clip_model, 'extract_features'):
                        layer_features = self.clip_model.extract_features(
                            dummy_image,
                            extract=[extract_layer]
                        )
                        layer_key = f'layer_{extract_layer}_cls'

                        if layer_key in layer_features:
                            sample_features = layer_features[layer_key]
                        else:
                            available_keys = list(layer_features.keys())
                            if available_keys:
                                sample_features = layer_features[available_keys[0]]
                                print(f"Warning: {layer_key} not found. Using {available_keys[0]} instead.")
                            else:
                                sample_features = self.clip_model.encode_image(dummy_image)
                    else:
                        sample_features = self.clip_model.encode_image(dummy_image)

                except Exception as e:
                    print(f"Warning: Failed to extract features from layer {extract_layer}: {e}. Using standard encode_image instead.")
                    sample_features = self.clip_model.encode_image(dummy_image)

            else:
                sample_features = self.clip_model.encode_image(dummy_image)

            self.feature_dim = sample_features.shape[-1]
            self.feature_dtype = sample_features.dtype

        print(f"CLIP feature dimension: {self.feature_dim}, dtype: {self.feature_dtype}")

        self.cache_keys = None
        self.cache_values = None

    # --------------------------------------
    # Feature extraction (intermediate layer supported)
    # --------------------------------------

    @torch.no_grad()
    def _extract_image_features(self, images):

        extract_layer = self.config.get("clip_layer", None)

        if extract_layer is not None and hasattr(self.clip_model, "extract_features"):

            layer_features = self.clip_model.extract_features(
                images,
                extract=[extract_layer]
            )

            layer_key = f'layer_{extract_layer}_cls'

            if layer_key in layer_features:
                features = layer_features[layer_key]
            else:
                available_keys = list(layer_features.keys())
                features = layer_features[available_keys[0]]

        else:
            features = self.clip_model.encode_image(images)

        features = F.normalize(features, dim=-1)

        return features

    # --------------------------------------
    # Cache construction
    # --------------------------------------

    def collect_cache_samples(self):

        image_paths = self.config["cache_images"]
        labels = self.config["cache_labels"]

        dataset = CacheDataset(image_paths, labels, self.preprocess)

        loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )

        all_features = []
        all_labels = []

        for images, lbls in tqdm(loader, desc="Building cache"):
            images = images.to(self.device)
            feats = self._extract_image_features(images)

            all_features.append(feats)
            all_labels.append(lbls)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return features, labels

    def build_cache_model(self, features, labels):

        self.cache_keys = features.t()

        self.cache_values = F.one_hot(
            labels,
            num_classes=self.config["num_classes"]
        ).float().to(self.device)

    # --------------------------------------
    # Inference
    # --------------------------------------

    def _predict_batch(self, images):

        features = self._extract_image_features(images)

        affinity = features @ self.cache_keys
        logits = torch.exp(self.config["beta"] * affinity) @ self.cache_values

        return logits

    # --------------------------------------
    # Evaluation
    # --------------------------------------

    def run_evaluation(self):

        print("Starting FTNet deepfake detection evaluation...")

        results = {}

        for name, dataset_cfg in self.config["test_datasets"].items():

            dataset = CacheDataset(
                dataset_cfg["images"],
                dataset_cfg["labels"],
                self.preprocess
            )

            loader = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn
            )

            all_preds = []
            all_probs = []
            all_labels = []

            for images, lbls in tqdm(loader, desc=f"Evaluating {name}"):
                images = images.to(self.device)

                logits = self._predict_batch(images)
                probs = F.softmax(logits, dim=-1)

                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(lbls.numpy())

            acc = accuracy_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)

            results[name] = {
                "accuracy": acc,
                "auc": auc
            }

        print("\nFTNet evaluation completed.")

        return {
            "method": "FTNet",
            "results": results
        }


# ==========================================
# Main
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description='FTNet for Deepfake Detection'
    )
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_random_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = FTNet(config)

    if model.cache_keys is None:
        cache_features, cache_labels = model.collect_cache_samples()
        model.build_cache_model(cache_features, cache_labels)

    results = model.run_evaluation()

    with open("ftnet_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

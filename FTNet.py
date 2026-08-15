#!/usr/bin/env python3
import os
import yaml
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

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
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        label = self.labels[idx]
        return image, label


# ==========================================
# FTNet
# ==========================================

class FTNet:

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_cfg = self.config["model"]
        requested_device = model_cfg.get("device", "cuda")
        self.device = requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip_load(
            model_cfg["backbone"], device=self.device,
            download_root=model_cfg.get("download_root")
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

        features = features.float()
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
            batch_size=self.config["training"]["batch_size"],
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
            num_classes=2
        ).float().to(self.device)

    # --------------------------------------
    # Inference
    # --------------------------------------

    def _predict_batch(self, images):

        features = self._extract_image_features(images)

        affinity = features @ self.cache_keys
        beta = self.config["hyperparams"]["init_beta"]
        logits = torch.exp(-beta + beta * affinity) @ self.cache_values

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
                batch_size=self.config["evaluation"]["batch_size"],
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

        mean_accuracy = float(np.mean([item["accuracy"] for item in results.values()]))
        mean_auc = float(np.mean([item["auc"] for item in results.values()]))
        print(f"Mean accuracy: {mean_accuracy:.4f}, mean AUC: {mean_auc:.4f}")

        return {
            "method": "FTNet",
            "summary": {
                "mean_accuracy": mean_accuracy,
                "mean_auc": mean_auc
            },
            "results": results
        }


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _class_images(dataset_dirs, class_name):
    paths = []
    for dataset_dir in dataset_dirs:
        class_dir = Path(dataset_dir) / class_name
        if class_dir.is_dir():
            paths.extend(
                str(path) for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(paths)


def prepare_few_shot_config(config, shots, seed, max_test_per_class=None):
    """Discover GenImage folders and make a leakage-free per-dataset split."""
    root = Path(config["data"]["test_data_root"] or "")
    if not root.is_dir():
        raise ValueError(f"Invalid data.test_data_root: {root}")

    rng = random.Random(seed)
    cache_images, cache_labels = [], []
    test_datasets = {}
    sd_parts = ["stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "wukong"]

    for name in config["data"]["datasets"]:
        parts = sd_parts if name.lower() == "sd" else [name]
        dirs = [root / part for part in parts]
        # dirs = [root / name]
        images, labels = [], []
        for label, class_name in enumerate(("0_real", "1_fake")):
            candidates = _class_images(dirs, class_name)
            if len(candidates) <= shots:
                raise ValueError(
                    f"{name}/{class_name} has {len(candidates)} images; need more than {shots}"
                )
            selected = set(rng.sample(candidates, shots))
            cache_images.extend(sorted(selected))
            cache_labels.extend([label] * shots)
            remaining = [path for path in candidates if path not in selected]
            if max_test_per_class and len(remaining) > max_test_per_class:
                remaining = rng.sample(remaining, max_test_per_class)
            images.extend(remaining)
            labels.extend([label] * len(remaining))
        test_datasets[name] = {"images": images, "labels": labels}
        print(f"{name}: cache={2 * shots}, test={len(images)}")

    config["cache_images"] = cache_images
    config["cache_labels"] = cache_labels
    config["test_datasets"] = test_datasets
    return config


# ==========================================
# Main
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description='FTNet for Deepfake Detection'
    )
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-beta', type=float, default=None)
    parser.add_argument('--shots', type=int, default=None,
                        help='Cache examples per class and dataset')
    parser.add_argument('--max-test-per-class', type=int, default=1000,
                        help='Optional evaluation cap per class for a smoke test')

    args = parser.parse_args()

    set_random_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.init_beta is not None:
        config["hyperparams"]["init_beta"] = args.init_beta

    shots = args.shots or config["cache"]["shots_per_class"]
    config = prepare_few_shot_config(config, shots, args.seed, args.max_test_per_class)

    model = FTNet(config)

    if model.cache_keys is None:
        cache_features, cache_labels = model.collect_cache_samples()
        model.build_cache_model(cache_features, cache_labels)

    results = model.run_evaluation()

    beta = float(config["hyperparams"]["init_beta"])
    results_file = config["evaluation"].get("results_file") or f"ftnet_{shots}shot_beta{beta:g}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()

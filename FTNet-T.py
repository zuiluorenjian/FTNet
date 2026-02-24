#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from networks.clip.clip import load as clip_load
from networks.clip.clip import tokenize as clip_tokenize


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels


class CacheDataset(Dataset):
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.preprocess(image)
            return image_tensor, label
        except Exception:
            return None, -1


class FTNet_T:

    def __init__(self, config):
        self.config = config
        self.device = config['model']['device']
        self.backbone = config['model']['backbone']
        self.download_root = config['model'].get('download_root', None)

        set_random_seed(40)

        self._load_clip_model()

        self.test_datasets = config['data']['datasets']

        self.cache_keys = None
        self.cache_values = None

        self.beta = config['hyperparams']['init_beta']
        self.alpha = config['hyperparams']['init_alpha']

        os.makedirs(config['cache']['cache_dir'], exist_ok=True)

    def _load_clip_model(self):
        if self.download_root:
            self.clip_model, self.preprocess = clip_load(
                self.backbone, device=self.device, download_root=self.download_root
            )
        else:
            self.clip_model, self.preprocess = clip_load(
                self.backbone, device=self.device
            )

        self.clip_model.eval()

        # ===== DO NOT MODIFY THIS BLOCK =====
        extract_layer = self.config.get('clip_layer', None)
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)

            if extract_layer is not None:
                try:
                    if hasattr(self.clip_model, 'extract_features'):
                        layer_features = self.clip_model.extract_features(
                            dummy_image, extract=[extract_layer]
                        )
                        layer_key = f'layer_{extract_layer}_cls'
                        if layer_key in layer_features:
                            sample_features = layer_features[layer_key]
                        else:
                            available_keys = list(layer_features.keys())
                            if available_keys:
                                sample_features = layer_features[available_keys[0]]
                                print(
                                    f"Warning: {layer_key} not found, using {available_keys[0]}"
                                )
                            else:
                                sample_features = self.clip_model.encode_image(
                                    dummy_image
                                )
                    else:
                        sample_features = self.clip_model.encode_image(dummy_image)
                except Exception as e:
                    print(
                        f"Warning: failed to get features from layer {extract_layer}: {e}, using default encoding"
                    )
                    sample_features = self.clip_model.encode_image(dummy_image)
            else:
                sample_features = self.clip_model.encode_image(dummy_image)

            self.feature_dim = sample_features.shape[-1]
            self.feature_dtype = sample_features.dtype

        print(
            f"CLIP feature dimension: {self.feature_dim}, data type: {self.feature_dtype}"
        )
        # ====================================

    def _extract_image_features(self, images):
        extract_layer = self.config.get('clip_layer', None)

        if extract_layer is not None and hasattr(self.clip_model, 'extract_features'):
            try:
                layer_features = self.clip_model.extract_features(
                    images, extract=[extract_layer]
                )
                layer_key = f'layer_{extract_layer}_cls'
                if layer_key in layer_features:
                    features = layer_features[layer_key]
                else:
                    available_keys = list(layer_features.keys())
                    features = layer_features[available_keys[0]]
            except Exception:
                features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.encode_image(images)

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def build_cache_model(self, cache_images, cache_labels):
        all_features = []

        with torch.no_grad():
            for img_path in tqdm(cache_images, desc="Extracting CLIP features"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    features = self._extract_image_features(input_tensor)
                    all_features.append(features)
                except Exception:
                    continue

        if not all_features:
            raise ValueError("No valid features extracted for cache model.")

        self.cache_keys = torch.cat(all_features, dim=0).t()
        self.cache_values = F.one_hot(
            torch.tensor(cache_labels), num_classes=2
        ).float().to(self.device)

    def create_adapter(self):
        adapter = nn.Linear(
            self.cache_keys.shape[0],
            self.cache_keys.shape[1],
            bias=False,
        ).to(self.device)

        adapter.weight = nn.Parameter(self.cache_keys.t())
        return adapter

    def run(self):
        print("FTNet-T initialized successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="FTNet-T for Deepfake Detection"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='tip_adapter_config.yaml',
        help='Path to the config file',
    )
    parser.add_argument(
        '--clip_layer',
        type=int,
        default=None,
        help='CLIP feature extraction layer (e.g., 12)',
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.clip_layer is not None:
        config['clip_layer'] = args.clip_layer

    model = FTNet_T(config)
    model.run()


if __name__ == '__main__':
    main()

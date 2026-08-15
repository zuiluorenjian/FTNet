#!/usr/bin/env python3
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.clip.clip import load as clip_load
from FTNet import CacheDataset, collate_fn, prepare_few_shot_config


def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FTNetT:
    def __init__(self, config, seed=42):
        self.config = config
        requested = config["model"].get("device", "cuda")
        self.device = requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        self.beta = float(config["hyperparams"]["init_beta"])
        set_random_seed(seed)
        self.clip_model, self.preprocess = clip_load(
            config["model"]["backbone"], device=self.device,
            download_root=config["model"].get("download_root"))
        self.clip_model.eval().requires_grad_(False)
        self.cache_keys = self.cache_values = self.adapter = None

    @torch.no_grad()
    def extract_features(self, images):
        layer = self.config.get("clip_layer")
        if layer is not None and hasattr(self.clip_model, "extract_features"):
            extracted = self.clip_model.extract_features(images, extract=[layer])
            key = f"layer_{layer}_cls"
            if key not in extracted:
                raise KeyError(f"Missing {key}; available keys: {list(extracted)}")
            features = extracted[key]
        else:
            features = self.clip_model.encode_image(images)
        return F.normalize(features.float(), dim=-1)

    def build_cache(self):
        dataset = CacheDataset(self.config["cache_images"], self.config["cache_labels"], self.preprocess)
        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_fn)
        features, labels = [], []
        for images, batch_labels in tqdm(loader, desc="Building trainable cache"):
            features.append(self.extract_features(images.to(self.device)))
            labels.append(batch_labels)
        cache_features = torch.cat(features)
        cache_labels = torch.cat(labels).to(self.device)
        self.cache_keys = cache_features.t().contiguous()
        self.cache_values = F.one_hot(cache_labels, num_classes=2).float()
        self.adapter = nn.Linear(cache_features.shape[1], cache_features.shape[0], bias=False)
        self.adapter = self.adapter.to(self.device, dtype=torch.float32)
        self.adapter.weight = nn.Parameter(cache_features.clone())
        print(f"Cache: {cache_features.shape[0]} samples, feature dim: {cache_features.shape[1]}, dtype: {cache_features.dtype}")

    def cache_logits(self, features):
        affinity = self.adapter(features)
        return torch.exp(-self.beta + self.beta * affinity) @ self.cache_values

    def train_adapter(self):
        dataset = CacheDataset(self.config["cache_images"], self.config["cache_labels"], self.preprocess)
        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"],
                            shuffle=True, collate_fn=collate_fn)
        epochs = int(self.config["training"]["epochs"])
        optimizer = torch.optim.AdamW(
            self.adapter.parameters(), lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.adapter.train()
        for epoch in range(epochs):
            total_loss = correct = total = 0
            for images, labels in loader:
                labels = labels.to(self.device)
                features = self.extract_features(images.to(self.device))
                logits = self.cache_logits(features)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * labels.numel()
                correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
            scheduler.step()
            print(f"Epoch {epoch + 1:02d}/{epochs}: loss={total_loss / total:.4f}, cache_acc={correct / total:.4f}, lr={scheduler.get_last_lr()[0]:.6g}")
        self.adapter.eval()

    @torch.no_grad()
    def evaluate(self):
        results = {}
        for name, dataset_config in self.config["test_datasets"].items():
            dataset = CacheDataset(dataset_config["images"], dataset_config["labels"], self.preprocess)
            loader = DataLoader(dataset, batch_size=self.config["evaluation"]["batch_size"],
                                shuffle=False, collate_fn=collate_fn)
            predictions, probabilities, labels = [], [], []
            for images, batch_labels in tqdm(loader, desc=f"Evaluating {name}"):
                logits = self.cache_logits(self.extract_features(images.to(self.device)))
                probs = F.softmax(logits, dim=1)
                predictions.extend(logits.argmax(1).cpu().tolist())
                probabilities.extend(probs[:, 1].cpu().tolist())
                labels.extend(batch_labels.tolist())
            results[name] = {
                "accuracy": float(accuracy_score(labels, predictions)),
                "ap": float(average_precision_score(labels, probabilities)),
                "auc": float(roc_auc_score(labels, probabilities)),
                "f1": float(f1_score(labels, predictions)),
            }
            print(f"{name}: {results[name]}")
        summary = {key: float(np.mean([metrics[key] for metrics in results.values()]))
                   for key in ("accuracy", "ap", "auc", "f1")}
        print(f"Mean: {summary}")
        return {"method": "FTNet-T", "summary": summary, "results": results}

    def run(self, shots):
        self.build_cache(); self.train_adapter(); results = self.evaluate()
        cache_dir = self.config["cache"]["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        train_cfg = self.config["training"]
        run_name = (f"{shots}shot_beta{self.beta:g}_ep{int(train_cfg['epochs'])}"
                    f"_lr{float(train_cfg['learning_rate']):g}_bs{int(train_cfg['batch_size'])}")
        model_path = os.path.join(cache_dir, f"ftnet_t_adapter_{run_name}.pt")
        torch.save(self.adapter.state_dict(), model_path)
        results_path = self.config["evaluation"].get("results_file") or f"ftnet_t_{run_name}_results.json"
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Adapter saved to {model_path}"); print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Trainable FTNet cache adapter")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--clip_layer", type=int, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-beta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.clip_layer is not None: config["clip_layer"] = args.clip_layer
    if args.init_beta is not None: config["hyperparams"]["init_beta"] = args.init_beta
    if args.epochs is not None: config["training"]["epochs"] = args.epochs
    if args.learning_rate is not None: config["training"]["learning_rate"] = args.learning_rate
    if args.batch_size is not None: config["training"]["batch_size"] = args.batch_size
    shots = args.shots or config["cache"]["shots_per_class"]
    config = prepare_few_shot_config(config, shots, args.seed, args.max_test_per_class)
    FTNetT(config, args.seed).run(shots)


if __name__ == "__main__":
    main()

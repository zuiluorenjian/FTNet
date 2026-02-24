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
        except Exception as e:
            # Silently return None for failed images
            return None, -1


class TipAdapterF:

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
        self.image_projection = None
        self.use_projection = False
        
        self.beta = config['hyperparams']['init_beta']
        self.alpha = config['hyperparams']['init_alpha']
        
        os.makedirs(config['cache']['cache_dir'], exist_ok=True)

    def _load_clip_model(self):
        if self.download_root:
            self.clip_model, self.preprocess = clip_load(
                self.backbone, device=self.device, download_root=self.download_root
            )
        else:
            self.clip_model, self.preprocess = clip_load(self.backbone, device=self.device)
        
        self.clip_model.eval()
        
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            sample_features = self._extract_image_features(dummy_image)
            self.feature_dim = sample_features.shape[-1]
            self.feature_dtype = sample_features.dtype
        
        self.extract_layer = self.config.get('clip_layer', None)

    def _extract_image_features(self, images):
        extract_layer = self.config.get('clip_layer', None)
        
        if extract_layer is not None and hasattr(self.clip_model, 'extract_features'):
            try:
                layer_features = self.clip_model.extract_features(images, extract=[extract_layer])
                layer_key = f'layer_{extract_layer}_cls'
                if layer_key in layer_features:
                    features = layer_features[layer_key]
                else:
                    available_keys = list(layer_features.keys())
                    if available_keys:
                        features = layer_features[available_keys[0]]
                    else:
                        raise ValueError(f"Could not extract features from layer {extract_layer}")
            except Exception:
                features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.encode_image(images)
        
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        return features
        
    def setup_text_encoder(self):
        with torch.no_grad():
            dummy_text = clip_tokenize(["a photo"]).to(self.device)
            text_features = self.clip_model.encode_text(dummy_text)
            self.text_dim = text_features.shape[-1]
        
        self.use_projection = False
        self.image_projection = None
        
    def collect_cache_samples(self):
        test_data_root = self.config['data']['test_data_root']
        shots_per_class = self.config['cache']['shots_per_class']
        multiclass_flags = self.config['data'].get('multiclass', [0] * len(self.test_datasets))
        
        cache_images = []
        cache_labels = []
        cache_dataset_info = []
        
        for idx, dataset_name in enumerate(self.test_datasets):
            dataset_path = os.path.join(test_data_root, dataset_name)
            if not os.path.exists(dataset_path):
                continue
            
            is_multiclass = multiclass_flags[idx] if idx < len(multiclass_flags) else 0
            
            if is_multiclass:
                self._collect_multiclass_samples(
                    dataset_path, dataset_name, shots_per_class,
                    cache_images, cache_labels, cache_dataset_info
                )
            else:
                self._collect_binary_samples(
                    dataset_path, dataset_name, shots_per_class,
                    cache_images, cache_labels, cache_dataset_info
                )
        
        return cache_images, cache_labels, cache_dataset_info

    def _collect_multiclass_samples(self, dataset_path, dataset_name, shots_per_class,
                                    cache_images, cache_labels, cache_dataset_info):
        class_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        all_real_files = []
        all_fake_files = []
        
        for class_dir in class_dirs:
            class_path = os.path.join(dataset_path, class_dir)
            subdirs = [d for d in os.listdir(class_path) 
                       if os.path.isdir(os.path.join(class_path, d))]
            
            real_dir, fake_dir = self._find_real_fake_dirs(subdirs)
            
            if real_dir and fake_dir:
                real_path = os.path.join(class_path, real_dir)
                real_files = self._get_image_files(real_path)
                all_real_files.extend(real_files)
                
                fake_path = os.path.join(class_path, fake_dir)
                fake_files = self._get_image_files(fake_path)
                all_fake_files.extend(fake_files)
        
        if all_real_files:
            real_samples = random.sample(all_real_files, min(shots_per_class, len(all_real_files)))
            for img_path in real_samples:
                cache_images.append(img_path)
                cache_labels.append(0)
                cache_dataset_info.append(f"{dataset_name}_real")
        
        if all_fake_files:
            fake_samples = random.sample(all_fake_files, min(shots_per_class, len(all_fake_files)))
            for img_path in fake_samples:
                cache_images.append(img_path)
                cache_labels.append(1)
                cache_dataset_info.append(f"{dataset_name}_fake")

    def _collect_binary_samples(self, dataset_path, dataset_name, shots_per_class,
                                cache_images, cache_labels, cache_dataset_info):
        subdirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
        
        real_dir, fake_dir = self._find_real_fake_dirs(subdirs)
        
        if real_dir and fake_dir:
            real_path = os.path.join(dataset_path, real_dir)
            real_files = self._get_image_files(real_path)
            
            if real_files:
                real_samples = random.sample(real_files, min(shots_per_class, len(real_files)))
                for img_path in real_samples:
                    cache_images.append(img_path)
                    cache_labels.append(0)
                    cache_dataset_info.append(f"{dataset_name}_real")
            
            fake_path = os.path.join(dataset_path, fake_dir)
            fake_files = self._get_image_files(fake_path)
            
            if fake_files:
                fake_samples = random.sample(fake_files, min(shots_per_class, len(fake_files)))
                for img_path in fake_samples:
                    cache_images.append(img_path)
                    cache_labels.append(1)
                    cache_dataset_info.append(f"{dataset_name}_fake")

    def _find_real_fake_dirs(self, subdirs):
        real_dir = None
        fake_dir = None
        
        for subdir in subdirs:
            if '0_real' in subdir.lower() or subdir.lower() == 'real':
                real_dir = subdir
            elif '1_fake' in subdir.lower() or subdir.lower() == 'fake':
                fake_dir = subdir
        
        if real_dir is None or fake_dir is None:
            sorted_subdirs = sorted(subdirs)
            if len(sorted_subdirs) >= 2:
                real_dir = sorted_subdirs[0]
                fake_dir = sorted_subdirs[1]
        
        return real_dir, fake_dir

    def _get_image_files(self, directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        files = glob.glob(os.path.join(directory, '*.*'))
        return [f for f in files if f.lower().endswith(image_extensions)]
    
    def build_cache_model(self, cache_images, cache_labels):
        all_features = []
        
        with torch.no_grad():
            for img_path in tqdm(cache_images, desc="Extracting CLIP Features"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    features = self._extract_image_features(input_tensor)
                    all_features.append(features)
                except Exception:
                    continue
        
        if all_features:
            self.cache_keys = torch.cat(all_features, dim=0).t()
            
            self.cache_values = F.one_hot(
                torch.tensor(cache_labels), num_classes=2
            ).float().to(self.device)
        else:
            raise ValueError("Could not build cache model, no valid features.")
    
    def create_adapter(self):
        adapter = nn.Linear(
            self.cache_keys.shape[0], 
            self.cache_keys.shape[1], 
            bias=False
        ).to(self.device)
        adapter.weight = nn.Parameter(self.cache_keys.t())
        return adapter
    
    def finetune_adapter(self, cache_images, cache_labels):
        train_dataset = CacheDataset(cache_images, cache_labels, self.preprocess)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['finetune']['batch_size'],
            shuffle=True, 
            num_workers=4, 
            collate_fn=collate_fn
        )
        
        adapter = self.create_adapter()
        
        params_to_train = list(adapter.parameters())
        
        optimizer = torch.optim.AdamW(
            params_to_train, 
            lr=self.config['finetune']['lr'], 
            eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.config['finetune']['epochs'] * len(train_loader)
        )
        
        beta, alpha = self.beta, self.alpha
        
        for epoch in range(self.config['finetune']['epochs']):
            adapter.train()
            total_loss = 0
            
            for images, labels in tqdm(train_loader, desc=f"Finetuning Epoch {epoch+1}/{self.config['finetune']['epochs']}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if images.shape[0] == 0:
                    continue
                
                with torch.no_grad():
                    self.clip_model.eval()
                    image_features_raw = self._extract_image_features(images)
                
                affinity = adapter(image_features_raw.to(adapter.weight.dtype))
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
                tip_logits = cache_logits
                loss = F.cross_entropy(tip_logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        adapter.eval()
        
        return adapter

    def _collect_test_samples(self, dataset_path, exclude_paths=None):
        test_images = []
        test_labels = []
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        is_multiclass = False
        for subdir in subdirs:
            class_path = os.path.join(dataset_path, subdir)
            if not os.path.isdir(class_path):
                continue
            subsubdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            if any(('real' in d.lower() or 'fake' in d.lower()) for d in subsubdirs):
                is_multiclass = True
                break

        if is_multiclass:
            for class_dir in subdirs:
                class_path = os.path.join(dataset_path, class_dir)
                if not os.path.isdir(class_path):
                    continue
                class_subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
                real_dir, fake_dir = self._find_real_fake_dirs(class_subdirs)
                if real_dir and fake_dir:
                    real_path = os.path.join(class_path, real_dir)
                    fake_path = os.path.join(class_path, fake_dir)
                    real_files = self._get_image_files(real_path)
                    fake_files = self._get_image_files(fake_path)
                    for img_path in real_files:
                        if exclude_paths and img_path in exclude_paths:
                            continue
                        test_images.append(img_path)
                        test_labels.append(0)
                    for img_path in fake_files:
                        if exclude_paths and img_path in exclude_paths:
                            continue
                        test_images.append(img_path)
                        test_labels.append(1)
        else:
            real_dir, fake_dir = self._find_real_fake_dirs(subdirs)
            if real_dir and fake_dir:
                real_path = os.path.join(dataset_path, real_dir)
                fake_path = os.path.join(dataset_path, fake_dir)
                real_files = self._get_image_files(real_path)
                fake_files = self._get_image_files(fake_path)
                for img_path in real_files:
                    if exclude_paths and img_path in exclude_paths:
                        continue
                    test_images.append(img_path)
                    test_labels.append(0)
                for img_path in fake_files:
                    if exclude_paths and img_path in exclude_paths:
                        continue
                    test_images.append(img_path)
                    test_labels.append(1)
        return test_images, test_labels
    
    def evaluate_all_datasets(self, adapter, cache_images=None):
        all_results = {}
        accuracies = []
        real_accuracies = []
        fake_accuracies = []
        ap_scores = []
        auc_scores = []
        f1_scores = []
        
        for dataset_name in self.test_datasets:
            try:
                accuracy, results = self._evaluate_single_dataset(dataset_name, adapter, exclude_paths=cache_images)
                all_results[dataset_name] = results
                accuracies.append(accuracy)
                real_accuracies.append(results.get('real_accuracy', 0.0))
                fake_accuracies.append(results.get('fake_accuracy', 0.0))
                ap_scores.append(results.get('ap_score', 0.0))
                auc_scores.append(results.get('auc_score', 0.0))
                f1_scores.append(results.get('f1_score', 0.0))
            except Exception as e:
                print(f"  Evaluation failed for {dataset_name}: {e}")
                accuracies.append(0.0)
                real_accuracies.append(0.0)
                fake_accuracies.append(0.0)
                ap_scores.append(0.0)
                auc_scores.append(0.0)
                f1_scores.append(0.0)
                all_results[dataset_name] = {}
        
        mean_accuracy = np.mean(accuracies) if accuracies else 0.0
        mean_real_accuracy = np.mean(real_accuracies) if real_accuracies else 0.0
        mean_fake_accuracy = np.mean(fake_accuracies) if fake_accuracies else 0.0
        mean_ap_score = np.mean(ap_scores) if ap_scores else 0.0
        mean_auc_score = np.mean(auc_scores) if auc_scores else 0.0
        mean_f1_score = np.mean(f1_scores) if f1_scores else 0.0
        
        self._print_results_table(accuracies, real_accuracies, fake_accuracies, 
                                  ap_scores, auc_scores, mean_accuracy, 
                                  mean_real_accuracy, mean_fake_accuracy, 
                                  mean_ap_score, mean_auc_score, f1_scores, mean_f1_score)
        
        return {
            'mean_accuracy': mean_accuracy,
            'mean_real_accuracy': mean_real_accuracy,
            'mean_fake_accuracy': mean_fake_accuracy,
            'mean_ap_score': mean_ap_score,
            'mean_auc_score': mean_auc_score,
            'mean_f1_score': mean_f1_score,
            'individual_results': all_results
        }
    
    def _evaluate_single_dataset(self, dataset_name, adapter, exclude_paths=None):
        test_data_root = self.config['data']['test_data_root']
        dataset_path = os.path.join(test_data_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        test_images, test_labels = self._collect_test_samples(dataset_path, exclude_paths=exclude_paths)
        
        if not test_images:
            raise ValueError("No test images found")
        
        all_predictions = []
        all_probabilities = []
        batch_size = self.config['evaluation']['batch_size']
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_images), batch_size), desc=f"Predicting on {dataset_name}"):
                batch_images_paths = test_images[i:i+batch_size]
                
                batch_tensors = []
                valid_indices_in_batch = []
                
                for j, img_path in enumerate(batch_images_paths):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        tensor = self.preprocess(image)
                        batch_tensors.append(tensor)
                        valid_indices_in_batch.append(j)
                    except:
                        continue
                
                if not batch_tensors:
                    continue
                
                batch_tensor = torch.stack(batch_tensors, dim=0).to(self.device)
                predictions, probabilities = self._predict_batch(batch_tensor, adapter)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Ensure labels match the number of successful predictions
        valid_labels = np.array(test_labels)[:len(all_predictions)]
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        if len(valid_labels) == 0:
             return 0.0, {}

        accuracy = (all_predictions == valid_labels).mean()
        
        real_mask = (valid_labels == 0)
        fake_mask = (valid_labels == 1)
        
        real_accuracy = (all_predictions[real_mask] == 0).mean() if real_mask.sum() > 0 else 0.0
        fake_accuracy = (all_predictions[fake_mask] == 1).mean() if fake_mask.sum() > 0 else 0.0
        
        fake_probs = all_probabilities[:, 1]
        ap_score = average_precision_score(valid_labels, fake_probs) if len(np.unique(valid_labels)) > 1 else 0.0
        auc_score = roc_auc_score(valid_labels, fake_probs) if len(np.unique(valid_labels)) > 1 else 0.0
        
        f1 = f1_score(valid_labels, all_predictions, average='binary')
        
        return accuracy, {
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'ap_score': ap_score,
            'auc_score': auc_score,
            'f1_score': f1,
            'total_samples': len(valid_labels),
            'real_samples': int(real_mask.sum()),
            'fake_samples': int(fake_mask.sum())
        }
    
    def _predict_batch(self, images, adapter):
        image_features = self._extract_image_features(images)
        
        affinity = adapter(image_features.to(adapter.weight.dtype))
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
        
        tip_logits = cache_logits
        probabilities = F.softmax(tip_logits, dim=1)
        predictions = tip_logits.argmax(dim=1)
        
        return predictions, probabilities
    
    def _print_results_table(self, accuracies, real_accuracies, fake_accuracies, 
                             ap_scores, auc_scores, mean_accuracy, mean_real_accuracy, 
                             mean_fake_accuracy, mean_ap_score, mean_auc_score, f1_scores, mean_f1_score):
        print(f"\n{'='*70}")
        print(f"{'Dataset':>15} {'Overall':>9} {'Real':>9} {'Fake':>9} {'AP':>8} {'AUC':>8} {'F1':>8}")
        print(f"{'='*70}")
        for i, dataset_name in enumerate(self.test_datasets):
            print(f"{dataset_name:>15} {accuracies[i]*100:>8.2f}% {real_accuracies[i]*100:>8.2f}% "
                  f"{fake_accuracies[i]*100:>8.2f}% {ap_scores[i]:>7.3f} {auc_scores[i]:>7.3f} {f1_scores[i]:>7.3f}")
        print(f"{'='*70}")
        print(f"{'Mean':>15} {mean_accuracy*100:>8.2f}% {mean_real_accuracy*100:>8.2f}% "
              f"{mean_fake_accuracy*100:>8.2f}% {mean_ap_score:>7.3f} {mean_auc_score:>7.3f} {mean_f1_score:>7.3f}")
        print(f"{'='*70}")
    
    def save_results(self, results, adapter_path):
        if not self.config['evaluation']['save_results']:
            return
        
        final_results = {
            'method': 'TIP-Adapter-F',
            'backbone': self.backbone,
            'config': self.config,
            'hyperparameters': {
                'beta': self.beta,
                'alpha': self.alpha
            },
            'results': results,
            'adapter_path': adapter_path,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = self.config['evaluation']['results_file']
        results_file = results_file.replace('.json', '_tip_adapter_f.json')
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
    
    def run(self):
        start_time = time.time()
        
        try:
            self.setup_text_encoder()
            
            cache_images, cache_labels, _ = self.collect_cache_samples()
            if not cache_images:
                raise ValueError("Failed to collect cache samples, please check data path and config.")
            
            self.build_cache_model(cache_images, cache_labels)
            
            adapter = self.finetune_adapter(cache_images, cache_labels)
            
            adapter_path = os.path.join(
                self.config['cache']['cache_dir'], 
                self.config['finetune']['adapter_weights']
            )
            torch.save(adapter.state_dict(), adapter_path)
            print(f"Adapter saved to: {adapter_path}")
            
            results = self.evaluate_all_datasets(adapter, cache_images=cache_images)
            
            self.save_results(results, adapter_path)
            
            end_time = time.time()
            print(f"\nTotal time: {end_time - start_time:.2f} seconds")
            if results:
                print(f"Mean accuracy: {results.get('mean_accuracy', 0.0)*100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"Run failed: {e}", file=sys.stderr)
            return None


def main():
    parser = argparse.ArgumentParser(description='TIP-Adapter-F for Deepfake Detection')
    parser.add_argument('--config', type=str, default='tip_adapter_config.yaml',
                        help='Path to the config file')
    parser.add_argument('--clip_layer', type=int, default=None,
                        help='CLIP feature extraction layer (e.g., 12), uses last layer if not specified')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.clip_layer is not None:
        config['clip_layer'] = args.clip_layer
    
    tip_adapter = TipAdapterF(config)
    tip_adapter.run()


if __name__ == '__main__':
    main()
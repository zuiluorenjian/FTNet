# FTNet for Few-Shot AI-Generated Image Detection

This repository implements a CLIP-based cache classifier for few-shot AI-generated image detection. It provides a training-free FTNet variant and a trainable cache-adapter variant, FTNet-T.

## Features

- Few-shot real/fake detection using CLIP image features
- Training-free cache inference with `FTNet.py`
- Trainable cache adapter with `FTNet-T.py`
- Support for the final CLIP representation or an intermediate transformer layer
- Per-dataset Accuracy and ROC AUC evaluation
- Cache/test separation to prevent image leakage
- Seeded cache and test sampling

## Project Structure

```text
FTNet/
|-- FTNet.py
|-- FTNet-T.py
|-- config.yaml
|-- run_4shot_2000.sh
|-- run_4shot_2000_train.sh
|-- networks/clip/
`-- README.md
```

## Requirements

```bash
pip install torch torchvision numpy pillow pyyaml tqdm scikit-learn
```

The code uses the local CLIP implementation under `networks/clip`. Set the pretrained checkpoint in `config.yaml`:

```yaml
model:
  backbone: /path/to/ViT-L-14.pt
  device: cuda
  download_root:
```

## Dataset Layout

Each configured dataset must contain two class directories:

```text
GenImage/
|-- ADM/
|   |-- 0_real/
|   `-- 1_fake/
|-- BigGAN/
|   |-- 0_real/
|   `-- 1_fake/
|-- glide/
|   |-- 0_real/
|   `-- 1_fake/
|-- Midjourney/
|   |-- 0_real/
|   `-- 1_fake/
|-- stable_diffusion_v_1_4/
|   |-- 0_real/
|   `-- 1_fake/
|-- stable_diffusion_v_1_5/
|   |-- 0_real/
|   `-- 1_fake/
|-- wukong/
|   |-- 0_real/
|   `-- 1_fake/
`-- VQDM/
    |-- 0_real/
    `-- 1_fake/
```

In the current `FTNet.py`, the logical `SD` entry is constructed at runtime from `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5` and `wukong`. Each of these three source directories must contain `0_real` and `1_fake`. If you instead use a physical merged `SD` directory, replace the special `sd_parts` mapping with `dirs = [root / name]`.

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` and `.tiff`.

Configure the dataset root and evaluated subsets in `config.yaml`:

```yaml
data:
  test_data_root: /path/to/GenImage
  datasets: [ADM, BigGAN, glide, Midjourney, SD, VQDM]
```

## Training-Free FTNet

`FTNet.py` extracts normalized CLIP features from the few-shot cache and predicts each test image from its cache affinity:

```text
affinity = image_features @ cache_keys
cache_logits = exp(-beta + beta * affinity) @ cache_values
```

Run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python FTNet.py \
  --config config.yaml \
  --shots 4 \
  --max-test-per-class 2000 \
  --init-beta 15
```

Arguments:

- `--shots`: number of cache images per class and dataset
- `--max-test-per-class`: maximum test images sampled from each class of each dataset
- `--seed`: random seed used for cache and test sampling
- `--init-beta`: cache-affinity sharpness
- `--config`: YAML configuration path

Results are written to:

```text
ftnet_<shots>shot_beta<beta>_results.json
```

### Bash runner

Update the Conda path, environment name, repository path and GPU selection in the script for your machine.

```bash
bash run_4shot_2000.sh
bash run_4shot_2000.sh 15
```

The optional first argument overrides beta. The effective test limit is determined by `--max-test-per-class` inside the script, not by its filename. The current script uses `20000`.

## Trainable FTNet-T

`FTNet-T.py` initializes a linear adapter from the cache features and optimizes it on the few-shot cache while keeping CLIP frozen.

```bash
CUDA_VISIBLE_DEVICES=0 python FTNet-T.py \
  --config config.yaml \
  --shots 4 \
  --max-test-per-class 2000 \
  --init-beta 4 \
  --epochs 20 \
  --learning-rate 0.002 \
  --batch-size 32
```

Or use:

```bash
bash run_4shot_2000_train.sh 4 20 0.002 32
```

The positional arguments are:

```text
<beta> <epochs> <learning-rate> <batch-size>
```

FTNet-T saves the trained adapter under `cache.cache_dir`. Its result filename contains the shots, beta, epochs, learning rate and batch size.

## Evaluation Metrics

FTNet reports per-dataset and mean Accuracy and ROC AUC. FTNet-T additionally reports Average Precision and F1 score.

## Reproducibility Notes

The current `FTNet.py` uses one seeded random-number generator for both cache selection and test subsampling. Repeating exactly the same command with an unchanged dataset produces the same split. However, changing `--max-test-per-class` advances that generator differently and can change cache samples selected for subsequent classes and datasets.

For fair comparisons across different test-set limits:

1. keep and reuse a fixed cache image list; or
2. use independent random-number generators for cache and test selection.

Few-shot performance can vary substantially across seeds because only a small number of cache images represent each class. Formal experiments should report the mean and standard deviation over multiple seeds.

Keep these settings fixed when comparing runs:

- dataset contents and directory layout
- random seed
- cache shots
- CLIP checkpoint and feature layer
- beta
- test-set size

Result filenames do not contain the seed or test-set limit. Rename results or set `evaluation.results_file` to avoid overwriting runs.

## Configuration

Important fields in `config.yaml` include:

```yaml
clip_layer: 12

cache:
  shots_per_class: 4
  cache_dir: ./cache_deepfake

hyperparams:
  init_beta: 1.0

training:
  epochs: 20
  learning_rate: 0.001
  batch_size: 32
  weight_decay: 1e-4

evaluation:
  batch_size: 64
  results_file: ''
```

When `evaluation.results_file` is empty, the program generates the result filename automatically.

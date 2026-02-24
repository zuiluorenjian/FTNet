
This project implements the FTNet method based on CLIP for deepfake detection, with the following features:

- **Efficiency**: FTNet method, no need to fine-tune the pre-trained model
- **Flexibility**: Supports various CLIP models and feature extraction layers
- **Scalability**: Supports both multi-class and binary dataset structures

## 🚀 Main Features

### 1. Fast Validation Mode (`FTNet.py`)
- Uses only the cache classifier, no text classifier required
- Automatically excludes cache samples to avoid data leakage
- Provides detailed evaluation metrics (accuracy, AP, AUC, etc.)

### 2. Full Training Mode (`FTNet-T.py`)
- Supports joint training of cache and text classifiers
- Provides fine-tuning functionality for further performance improvement
- Supports t-SNE visualization analysis

## 📁 Project Structure

```
├── FTNet.py                    # Fast validation main script
├── FTNet-T.py                  # Full training main script
├── config.yaml                 # Configuration file
├── clip/                       # CLIP model directory
└── README.md                   # Project documentation
```

## 🛠️ Requirements

### Basic Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install tqdm
pip install pyyaml
pip install pillow
pip install numpy
pip install matplotlib
pip install seaborn
```

### CLIP Model
Make sure the CLIP model files are in the `clip/` directory, or specify the download path via the `download_root` parameter.

## 📊 Dataset Format

### Binary Classification Dataset Structure
```
dataset_name/
├── 0_real/          # Real images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── 1_fake/          # Fake images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Multi-class Dataset Structure
```
dataset_name/
├── class1/
│   ├── 0_real/      # Real images
│   └── 1_fake/      # Fake images
├── class2/
│   ├── 0_real/
│   └── 1_fake/
└── ...
```

## 🚀 Usage

```bash
# Use default config
python FTNet.py --config config.yaml

# Specify CLIP feature extraction layer
python FTNet-T.py --config config.yaml --clip_layer 12

# Use a specific GPU
CUDA_VISIBLE_DEVICES=0 python FTNet.py --config config.yaml
```


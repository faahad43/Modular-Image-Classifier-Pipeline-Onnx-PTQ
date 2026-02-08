# Modular Image Classification Training + Deployment (ONNX + PTQ)

A complete, modular PyTorch training and evaluation pipeline for image classification with ONNX export and Post-Training Quantization (PTQ).

## Features

### Part 1: Training + Evaluation Framework

✅ **Swappable Backbones**: Support for 4 different architectures
- Custom CNN (tiny architecture)
- ConvNeXt Tiny
- Vision Transformer (ViT-B/32)
- Faster R-CNN Backbone (ResNet50+FPN adapted for classification)

✅ **Swappable Datasets**: Easy dataset switching
- CIFAR-10
- CIFAR-100
- STL-10

✅ **Advanced Training Features** (toggleable via CLI):
- `--use_amp`: Automatic Mixed Precision training
- `--use_scheduler`: Cosine annealing learning rate scheduler
- `--use_grad_clip`: Gradient clipping (max_norm=1.0)
- `--use_weight_decay`: AdamW optimizer with weight decay
- `--freeze_backbone`: Transfer learning mode
- `--seed`: Reproducibility with fixed random seed

✅ **Comprehensive Evaluation**:
- Per-class metrics (accuracy, precision, recall, F1)
- Global metrics (macro avg, weighted avg)
- Confusion matrix plots
- Results saved to `part_1_results/`

### Part 2: ONNX Export + PTQ Comparison

✅ **Model Deployment Pipeline**:
- ONNX FP32 export
- Dynamic INT8 quantization
- Static INT8 quantization (with calibration)
- Comprehensive comparison report in `part_2_results/`

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch (with torchvision)
- ONNX Runtime
- scikit-learn
- pandas
- matplotlib
- seaborn

## Quick Start

### 1. Training

Train a model with specific architecture and dataset:

```bash
# Basic training (ConvNeXt on CIFAR-10)
python train.py --arch convnext_tiny --dataset cifar10 --epochs 10

# Training with all advanced features
python train.py \
  --arch convnext_tiny \
  --dataset cifar10 \
  --epochs 20 \
  --batch_size 32 \
  --use_amp \
  --use_scheduler \
  --use_grad_clip \
  --use_weight_decay \
  --freeze_backbone \
  --pretrained \
  --seed 42
```

**Available architectures**: `convnext_tiny`, `vit_b_32`, `faster_rcnn`, `custom_cnn`  
**Available datasets**: `cifar10`, `cifar100`, `stl10`

### 2. Evaluation

Evaluate a trained model and generate metrics + confusion matrix:

```bash
python eval.py \
  --arch convnext_tiny \
  --dataset cifar10 \
  --ckpt best.pt
```

Outputs:
- Confusion matrix: `part_1_results/cn_convnext_tiny_cifar10.png`
- Summary: `part_1_results/summary.txt`

### 3. ONNX Export

Export trained PyTorch model to ONNX format:

```bash
python export_onnx.py \
  --arch convnext_tiny \
  --ckpt best.pt \
  --num_classes 10 \
  --image_size 224 \
  --output model_fp32.onnx
```

### 4. Quantization

#### Dynamic INT8 Quantization

```bash
python quantize_dynamic.py
```

Outputs: `model_dynamic_quant.onnx`

#### Static INT8 Quantization (with calibration)

```bash
python quantize_static.py
```

Outputs: `static_model_quant.onnx`

### 5. ONNX Model Comparison

Compare all model variants (PyTorch FP32, ONNX FP32, Dynamic INT8, Static INT8):

```bash
python eval_onnx.py \
  --arch convnext_tiny \
  --dataset cifar10 \
  --ckpt best.pt \
  --onnx_fp32 model_fp32.onnx \
  --onnx_dynamic model_dynamic_quant.onnx \
  --onnx_static static_model_quant.onnx
```

Outputs:
- Comparison table: `part_2_results/summary.txt`
- Metrics: Accuracy, Macro F1, Weighted F1
- Accuracy drop analysis

## Project Structure

```
Modular-Image-Classifer-Pipeline-Onnx-PTQ/
├── train.py              # Training pipeline with toggleable features
├── eval.py               # Evaluation with metrics and confusion matrix
├── export_onnx.py        # ONNX export script
├── quantize_dynamic.py   # Dynamic quantization
├── quantize_static.py    # Static quantization with calibration
├── eval_onnx.py          # ONNX model comparison
├── models.py             # Model factory (4 architectures)
├── datasets.py           # Dataset factory (3 datasets)
├── transform.py          # Image transformations
├── requirements.txt      # Python dependencies
├── test.py               # run the best.py model for uploading image
├── best.pt               # Trained model checkpoint
├── part_1_results/       # Training evaluation results
│   ├── summary.txt
│   └── cn_*.png          # Confusion matrices
└── part_2_results/       # ONNX comparison results
    └── summary.txt
```

## Training Toggles Explained

| Argument | Description | Benefit |
|----------|-------------|---------|
| `--use_amp` | Mixed precision (FP16/FP32) training | ~2x faster training, less memory |
| `--use_scheduler` | Cosine annealing LR scheduler | Better convergence |
| `--use_grad_clip` | Gradient clipping (max_norm=1.0) | Prevents exploding gradients |
| `--use_weight_decay` | AdamW optimizer with regularization | Reduces overfitting |
| `--freeze_backbone` | Transfer learning mode | Faster training for pretrained models |
| `--seed` | Fixed random seed | Reproducible results |

## Example Workflows

### Experiment 1: Custom CNN on CIFAR-10

```bash
# Train
python train.py --arch custom_cnn --dataset cifar10 --epochs 30 --seed 42

# Evaluate
python eval.py --arch custom_cnn --dataset cifar10 --ckpt best.pt
```

### Experiment 2: ViT with all features on CIFAR-100

```bash
# Train
python train.py \
  --arch vit_b_32 \
  --dataset cifar100 \
  --epochs 20 \
  --use_amp \
  --use_scheduler \
  --use_grad_clip \
  --use_weight_decay \
  --pretrained \
  --freeze_backbone \
  --seed 42

# Evaluate
python eval.py --arch vit_b_32 --dataset cifar100 --ckpt best.pt
```

### Experiment 3: Complete ONNX Deployment Pipeline

```bash
# 1. Train model
python train.py --arch convnext_tiny --dataset cifar10 --epochs 10 --pretrained --freeze_backbone

# 2. Export to ONNX
python export_onnx.py --arch convnext_tiny --ckpt best.pt --num_classes 10

# 3. Quantize (dynamic)
python quantize_dynamic.py

# 4. Quantize (static)
python quantize_static.py

# 5. Compare all variants
python eval_onnx.py --arch convnext_tiny --dataset cifar10
```

## Results Format

### Part 1: Training Evaluation

`part_1_results/summary.txt`:
```
Experiment: convnext_tiny | Dataset: cifar10

Class-wise Metrics:
              precision  recall  f1-score  support
Class_0          0.85     0.87      0.86      1000
...

Global Metrics:
              precision  recall  f1-score  support
macro avg        0.84     0.85      0.84     10000
weighted avg     0.84     0.85      0.84     10000
```

### Part 2: ONNX Comparison

`part_2_results/summary.txt`:
```
Model Artifact          Accuracy  Macro F1  Weighted F1
PyTorch FP32              0.850     0.840        0.845
ONNX FP32                 0.850     0.840        0.845
ONNX INT8 Dynamic         0.847     0.837        0.842
ONNX INT8 Static          0.848     0.838        0.843
```

## Notes

- All models are trained on 224x224 images (ImageNet size)
- ImageNet normalization is used: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Best model is automatically saved as `best.pt` based on validation accuracy
- CUDA is automatically used if available, otherwise CPU

## Author

Syed Fahad Shah

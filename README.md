# 🌊 BlueGuard: Ocean Waste Classification

Binary image classification model to detect ocean waste using **EfficientNet-B0** and **PyTorch**.  
Fine-tuned on the BlueGuard dataset with dual GPU (T4x2) support via `DataParallel`.

---

## 📊 Results

| Metric    | ocean_clean | ocean_trash | Overall  |
|-----------|-------------|-------------|----------|
| Precision | 1.00        | 1.00        | 1.00     |
| Recall    | 1.00        | 1.00        | 1.00     |
| F1-Score  | 1.00        | 1.00        | 1.00     |
| Val Acc   | —           | —           | **99.87%** |

---

## 📁 Dataset

**BlueGuard: Ocean Waste Classification** via Kaggle

```
DATASETS/
└── BlueGuard: Ocean Waste Classification/
    ├── ocean_clean/
    └── ocean_trash/
```

The dataset is split programmatically at runtime: **80% train / 10% val / 10% test**

---

## 🧠 Model

| Component   | Detail                              |
|-------------|-------------------------------------|
| Backbone    | EfficientNet-B0 (ImageNet pretrained) |
| Classifier  | Dropout(0.3) → Linear(1280 → 2)    |
| Optimizer   | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler   | CosineAnnealingLR (T_max=20)       |
| Loss        | CrossEntropyLoss                    |
| Multi-GPU   | nn.DataParallel (T4x2)             |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/goktani/blueguard-ocean-waste-classification.git
cd blueguard-ocean-waste-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run on Kaggle

- Import this repository into a Kaggle notebook
- Attach the **BlueGuard: Ocean Waste Classification** dataset
- Enable **GPU T4 x2** accelerator
- Run all cells in order

---

## 📂 Project Structure

```
blueguard-ocean-waste-classification/
├── blueguard_classification.ipynb   # Main notebook
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Configuration

All hyperparameters are controlled via the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    "img_size"    : 224,
    "batch_size"  : 64,
    "epochs"      : 20,
    "lr"          : 1e-4,
    "num_classes" : 2,
    "train_ratio" : 0.80,
    "val_ratio"   : 0.10,
    "test_ratio"  : 0.10,
}
```

---

## 🔍 Potential Improvements

- **GradCAM** — visualize which regions influence the model's predictions
- **Larger backbone** — EfficientNet-B3 or Vision Transformer (ViT)
- **Test-Time Augmentation (TTA)** — improve robustness at inference
- **Deployment** — integrate into drone or satellite imagery pipelines

---

## 📜 License

This project is licensed under the MIT License.

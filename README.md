# Drug Side Effect Prediction with Transformer

Dự án dự đoán tác dụng phụ của thuốc sử dụng mô hình Transformer.

## Cấu trúc Project

```
drug-side-effect-prediction/
├── data/                           # Dữ liệu
│   ├── raw/                        # Dữ liệu gốc
│   │   ├── drug_SMILES_750.csv
│   │   ├── drug_codes_chembl_freq_1500.txt
│   │   └── subword_units_map_chembl_freq_1500.csv
│   ├── processed/                  # Dữ liệu đã xử lý
│   │   ├── SE_sub_index_50.npy
│   │   └── SE_sub_mask_50.npy
│   └── cache/                      # Cache cho dataloader
│
├── config.py                       # Cấu hình toàn bộ
├── dataset.py                      # Dataset classes (tối ưu PyTorch 2)
├── encoder.py                      # Transformer Encoder
├── model.py                        # Main model
├── smiles_encoder.py               # SMILES encoding utilities
├── preprocessing.py                # Data preprocessing
├── trainer.py                      # Training loop (tối ưu PyTorch 2)
├── evaluator.py                    # Evaluation
├── losses.py                       # Loss functions
├── metrics.py                      # Evaluation metrics
├── utils.py                        # Utilities
│
├── train.py                        # Training script
├── evaluate.py                     # Evaluation script
├── preprocess_data.py              # Data preprocessing script
│
├── checkpoints/                    # Model checkpoints
├── logs/                           # Training logs (TensorBoard)
├── results/                        # Kết quả dự đoán
│
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## Tối ưu hóa PyTorch 2.x

### 1. torch.compile()
```python
model = DrugSideEffectModel()
model = torch.compile(model, mode='reduce-overhead')
```

### 2. Mixed Precision Training với torch.amp
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(drug, se, drug_mask, se_mask)
    loss = criterion(output, target)
```

### 3. Optimized DataLoader
```python
DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,  # PyTorch 2.x feature
    prefetch_factor=2
)
```

### 4. Flash Attention (nếu có GPU hỗ trợ)
```python
# Sử dụng F.scaled_dot_product_attention thay vì attention thủ công
attention_output = F.scaled_dot_product_attention(
    query, key, value, attn_mask=mask
)
```

### 5. Gradient Checkpointing (tiết kiệm memory)
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(self.encoder_layer, x, mask)
```

### 6. Fused Optimizer (AdamW fused trong PyTorch 2)
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    fused=True  # Faster trên CUDA
)
```

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Sử dụng

### 1. Tiền xử lý dữ liệu
```bash
python preprocess_data.py
```

### 2. Training
```bash
# Training cơ bản
python train.py

# Training với custom config
python train.py --lr 1e-4 --batch_size 128 --epochs 200 --device cuda

# Training với mixed precision
python train.py --use_amp --compile_model
```

### 3. Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Dataset

- **Drugs**: 750 thuốc với SMILES representations
- **Side Effects**: 994 loại tác dụng phụ
- **Subword Units**: 2586 subword units cho SMILES encoding
- **Training Strategy**: 10-fold Stratified Cross-Validation

## Model Architecture

```
Input: Drug SMILES + Side Effect
         ↓
    Embeddings (Word + Position)
         ↓
┌────────────────┬────────────────┐
│  Drug Encoder  │  SE Encoder    │
│  (Transformer) │  (Transformer) │
│   8 layers     │   8 layers     │
└────────────────┴────────────────┘
         ↓
   Cross-Attention (Optional)
         ↓
   Interaction Layer
   (Outer Product)
    [50 x 50 x 200]
         ↓
     Sum → [50 x 50]
         ↓
    Conv2D (1→3)
         ↓
    Flatten → [6912]
         ↓
       MLP Decoder
    (6912→512→64→32→1)
         ↓
      Prediction
```

### Key Components:
- **Subword Encoding**: BPE encoding cho SMILES strings
- **Transformer Encoder**: 8 layers, 8 heads, dim=200
- **Interaction**: Outer product + CNN
- **Decoder**: Deep MLP với batch normalization

## Metrics

- **Binary Classification**: AUC-ROC, AUPR, Precision, Recall, Accuracy
- **Regression**: RMSE, MAE, Pearson, Spearman correlation

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (optional, for GPU)

## License

MIT License
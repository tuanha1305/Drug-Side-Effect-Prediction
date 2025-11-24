# Drug Side Effect Prediction with Transformer

Dự án dự đoán tác dụng phụ của thuốc sử dụng mô hình Transformer.

## 2. Training
```bash
# Training cơ bản
python train.py

# Training với custom config
python train.py --lr 1e-4 --batch_size 128 --epochs 200 --device cuda

# Training với mixed precision
python train.py --use_amp --compile_model
```

## 3. Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Dataset

- **Drugs**: 750 thuốc với SMILES representations
- **Side Effects**: 994 loại tác dụng phụ
- **Subword Units**: 2586 subword units cho SMILES encoding
- **Training Strategy**: 10-fold Stratified Cross-Validation

### Key Components:
- **Subword Encoding**: BPE encoding cho SMILES strings
- **Transformer Encoder**: 8 layers, 8 heads, dim=200
- **Interaction**: Outer product + CNN
- **Decoder**: Deep MLP với batch normalization

## Metrics (from Paper)

**Primary Metrics (Regression Task):**
- **RMSE** - Root Mean Squared Error (frequency prediction)
- **MAE** - Mean Absolute Error (frequency prediction)
- **SCC** - Spearman's rank correlation coefficient (association prediction)
- **Overlap@N%** - Recommendation metrics (1%, 5%, 10%, 20% for ranking evaluation)

**Additional Metrics (Supplementary):**
- Pearson correlation
- Per-drug RMSE/MAE
- Binary analysis (for optional visualization only, not in paper)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (optional, for GPU)

## License

MIT License
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
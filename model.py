"""
Main model for drug side effect prediction
Updated with Bidirectional Cross-Attention and Residual Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from encoder import Embeddings, Encoder_MultipleLayers
from config import ModelConfig

# ============================================================================
# New Modules: Fusion & Cross-Attention
# ============================================================================

class ResidualFusion(nn.Module):
    """
    Residual Fusion Layer: Fuses original embedding with cross-attention output.
    Formula: LayerNorm(E + Dropout(CA_out))
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, E, CA_out):
        """
        Args:
            E: Original embedding (Batch, Seq_Len, Hidden)
            CA_out: Cross-Attention output (Batch, Seq_Len, Hidden)
        """
        # Residual connection: Original + Attention Info
        out = E + self.dropout(CA_out)
        # Normalize
        out = self.ln(out)
        return out

class GatedFusion(nn.Module):
    """
    Alternative: Gated Fusion Layer (Optional use)
    Learns a gate to decide how much context to accept.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, E, CA_out):
        # Concatenate along hidden dimension
        x = torch.cat([E, CA_out], dim=-1)
        # Calculate gate (0 to 1)
        g = torch.sigmoid(self.gate(x))
        # Fused output
        out = g * CA_out + (1 - g) * E
        return self.ln(out)

class BidirectionalCrossAttention(nn.Module):
    """
    2-way Cross Attention Module.
    Computes:
    1. Drug attending to Side Effect (How relevant is SE to this Drug part?)
    2. Side Effect attending to Drug (How relevant is Drug to this SE part?)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

        # Using Standard PyTorch MultiheadAttention for robust cross-attention
        self.mha_d_to_s = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.mha_s_to_d = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

    def forward(self, drug_emb, se_emb, drug_mask, se_mask):
        """
        Args:
            drug_emb: (Batch, D_Len, Hidden)
            se_emb: (Batch, S_Len, Hidden)
            drug_mask: (Batch, D_Len) - 1 for valid, 0 for padding
            se_mask: (Batch, S_Len)
        Returns:
            ca_d: Context aware drug embeddings
            ca_s: Context aware se embeddings
        """
        # Prepare masks for PyTorch MHA (True = Ignored/Padding)
        # Invert our mask: (1 -> False/Keep, 0 -> True/Ignore)
        key_padding_mask_drug = (drug_mask == 0)
        key_padding_mask_se = (se_mask == 0)

        # 1. Drug attends to Side Effect (Query=Drug, Key=SE, Value=SE)
        # "What parts of the Side Effect are relevant to this Drug substructure?"
        ca_d, _ = self.mha_d_to_s(
            query=drug_emb,
            key=se_emb,
            value=se_emb,
            key_padding_mask=key_padding_mask_se # Masking keys (SE)
        )

        # 2. Side Effect attends to Drug (Query=SE, Key=Drug, Value=Drug)
        # "What parts of the Drug are relevant to this Side Effect substructure?"
        ca_s, _ = self.mha_s_to_d(
            query=se_emb,
            key=drug_emb,
            value=drug_emb,
            key_padding_mask=key_padding_mask_drug # Masking keys (Drug)
        )

        return ca_d, ca_s

# ============================================================================
# Main Model
# ============================================================================

class DrugSideEffectModel(nn.Module):
    """
    Transformer-based model for drug side effect prediction
    Architecture:
        1. Independent Encoders (Drug & SE)
        2. Cross-Attention & Fusion (Optional/Configurable)
        3. Scalar Projection (Interaction Map)
        4. CNN -> MLP -> Score
    """

    def __init__(self, config: ModelConfig, device: str = 'cpu'):
        super(DrugSideEffectModel, self).__init__()

        self.config = config
        self.device = device

        # === Embedding layers ===
        self.emb_drug = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.embedding_dim,
            max_position_size=config.max_position_embeddings,
            dropout_rate=config.dropout_rate
        )

        self.emb_side = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.embedding_dim,
            max_position_size=config.max_position_embeddings,
            dropout_rate=config.dropout_rate
        )

        # === Transformer encoders (Self-Attention) ===
        self.encoder_drug = Encoder_MultipleLayers(
            n_layer=config.num_encoder_layers,
            hidden_size=config.embedding_dim,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_dropout,
            hidden_dropout_prob=config.hidden_dropout,
            use_flash_attention=config.use_flash_attention,
            use_sdpa=config.use_sdpa,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )

        self.encoder_side = Encoder_MultipleLayers(
            n_layer=config.num_encoder_layers,
            hidden_size=config.embedding_dim,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_dropout,
            hidden_dropout_prob=config.hidden_dropout,
            use_flash_attention=config.use_flash_attention,
            use_sdpa=config.use_sdpa,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )

        # === Cross-Attention & Fusion (New) ===
        self.use_cross_attention = config.use_cross_attention
        if self.use_cross_attention:
            # 1. Bidirectional Cross Attention
            self.cross_attention = BidirectionalCrossAttention(config)

            # 2. Residual Fusion Blocks
            self.fusion_drug = ResidualFusion(config.embedding_dim, config.hidden_dropout)
            self.fusion_se = ResidualFusion(config.embedding_dim, config.hidden_dropout)

        # === Interaction Module (Scalar Projection + CNN) ===
        self.interaction_cnn = nn.Conv2d(
            in_channels=1,
            out_channels=config.conv_out_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_padding
        )

        # === Decoder (MLP) ===
        self.decoder = self._build_decoder(
            input_dim=config.decoder_input_dim,
            hidden_dims=config.decoder_hidden_dims,
            output_dim=config.decoder_output_dim,
            dropout=config.decoder_dropout,
            use_batch_norm=config.use_batch_norm
        )

        # Dropout for interaction map
        self.interaction_dropout = nn.Dropout(config.decoder_dropout)

    def _build_decoder(
            self, input_dim: int, hidden_dims: list, output_dim: int,
            dropout: float, use_batch_norm: bool
    ) -> nn.Sequential:
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Mask for custom Transformer Encoder (Self-Attention)"""
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -1e9
        return attention_mask

    def forward(
            self,
            drug: torch.Tensor,
            side_effect: torch.Tensor,
            drug_mask: torch.Tensor,
            se_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = drug.size(0)

        # Move to device
        drug = drug.long().to(self.device)
        side_effect = side_effect.long().to(self.device)
        drug_mask = drug_mask.long().to(self.device)
        se_mask = se_mask.long().to(self.device)

        # 1. Embedding & Self-Attention Encoding
        # --------------------------------------
        drug_emb = self.emb_drug(drug)
        se_emb = self.emb_side(side_effect)

        drug_encoded = self.encoder_drug(
            drug_emb.float(), self._create_attention_mask(drug_mask).float()
        )
        se_encoded = self.encoder_side(
            se_emb.float(), self._create_attention_mask(se_mask).float()
        )

        # 2. Cross-Attention & Fusion (If Enabled)
        # ----------------------------------------
        if self.use_cross_attention:
            # A. Calculate Cross-Attention
            ca_d, ca_s = self.cross_attention(
                drug_encoded, se_encoded, drug_mask, se_mask
            )

            # B. Residual Fusion: LayerNorm(Original + Dropout(CA))
            # These are the vectors that will form the interaction map
            drug_final = self.fusion_drug(drug_encoded, ca_d)
            se_final = self.fusion_se(se_encoded, ca_s)
        else:
            # If disabled, just use the self-attended features
            drug_final = drug_encoded
            se_final = se_encoded

        # 3. Interaction Module (Scalar Projection)
        # -----------------------------------------
        # Uses fused features if CA is enabled, or raw features otherwise
        drug_aug = drug_final.unsqueeze(2)  # (b, d, 1, c)
        se_aug = se_final.unsqueeze(1)      # (b, 1, s, c)

        # Dot product interaction
        interaction = drug_aug * se_aug
        interaction = interaction.permute(0, 3, 1, 2)  # (b, c, d, s)
        interaction_map = torch.sum(interaction, dim=1, keepdim=True)  # (b, 1, d, s)

        interaction_map = self.interaction_dropout(interaction_map)

        # 4. CNN & Prediction
        # -------------------
        interaction_features = self.interaction_cnn(interaction_map)
        interaction_flat = interaction_features.view(batch_size, -1)

        raw_score = self.decoder(interaction_flat)

        # Optional: Add ReLU to ensure non-negative score?
        # HSTrans paper uses Linear output, but logically frequency >= 0.
        # We stick to Linear to allow full gradient flow, clipping handled in Evaluator.
        score = raw_score

        return score, drug_final, se_final

    def count_parameters(self) -> dict:
        """Count trainable parameters"""
        counts = {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoders': sum(p.numel() for p in self.encoder_drug.parameters()) * 2,
            'decoder': sum(p.numel() for p in self.decoder.parameters())
        }
        if self.use_cross_attention:
            counts['cross_attention'] = sum(p.numel() for p in self.cross_attention.parameters())
            counts['fusion'] = sum(p.numel() for p in self.fusion_drug.parameters()) * 2
        return counts

def create_model(config: ModelConfig, device: str = 'cpu') -> DrugSideEffectModel:
    return DrugSideEffectModel(config, device).to(device)
"""
Main model for drug side effect prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from encoder import Embeddings, Encoder_MultipleLayers
from config import ModelConfig


class DrugSideEffectModel(nn.Module):
    """
    Transformer-based model for drug side effect prediction

    Architecture (following HSTrans paper):
        1. Drug Encoder (Transformer)
        2. Side Effect Encoder (Transformer)
        3. Interaction Module:
           - Scalar Projection Layer: I = E_d · E_s^T
           - CNN Layer: M = CNN(I)
        4. Decoder (MLP → raw logit output)
    """

    def __init__(self, config: ModelConfig, device: str = 'cpu'):
        super(DrugSideEffectModel, self).__init__()

        self.config = config
        self.device = device

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.decoder_dropout)

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

        # === Transformer encoders ===
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

        # === Interaction Module ===
        # Scalar projection layer is implemented in forward()
        # CNN layer to capture local region interactions
        self.interaction_cnn = nn.Conv2d(
            in_channels=1,  # Single channel interaction map
            out_channels=config.conv_out_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_padding
        )

        # === Decoder (MLP) ===
        # Calculate input dimension based on CNN output
        # After CNN: (batch, out_channels, d, s) where d and s depend on input size
        # We'll calculate this dynamically or use config
        self.decoder = self._build_decoder(
            input_dim=config.decoder_input_dim,
            hidden_dims=config.decoder_hidden_dims,
            output_dim=config.decoder_output_dim,
            dropout=config.decoder_dropout,
            use_batch_norm=config.use_batch_norm
        )

    def _build_decoder(
            self,
            input_dim: int,
            hidden_dims: list,
            output_dim: int,
            dropout: float,
            use_batch_norm: bool
    ) -> nn.Sequential:
        """Build MLP decoder with optional batch normalization"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(True))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final linear → raw logit output (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask in transformer format"""
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
        """
        Forward pass – returns raw logits

        Args:
            drug: (batch, max_drug_len) drug substructure indices
            side_effect: (batch, max_se_len) SE substructure indices
            drug_mask: (batch, max_drug_len) mask for drug
            se_mask: (batch, max_se_len) mask for SE

        Returns:
            score: (batch, 1) predicted frequency score (raw logits)
            drug_encoded: (batch, max_drug_len, embedding_dim)
            se_encoded: (batch, max_se_len, embedding_dim)
        """
        batch_size = drug.size(0)

        # Move to device
        drug = drug.long().to(self.device)
        side_effect = side_effect.long().to(self.device)
        drug_mask = drug_mask.long().to(self.device)
        se_mask = se_mask.long().to(self.device)

        # === Attention masks ===
        drug_attention_mask = self._create_attention_mask(drug_mask)
        se_attention_mask = self._create_attention_mask(se_mask)

        # === Embedding & Encoding ===
        # E_d^0 = E_d^c + E_d^p (content + position embeddings)
        drug_emb = self.emb_drug(drug)  # (batch, d, c)
        se_emb = self.emb_side(side_effect)  # (batch, s, c)

        # Transformer encoding: E_d and E_s
        drug_encoded = self.encoder_drug(
            drug_emb.float(), drug_attention_mask.float(), fusion=False
        )  # (batch, d, c)

        se_encoded = self.encoder_side(
            se_emb.float(), se_attention_mask.float(), fusion=False
        )  # (batch, s, c)

        # ===================================================================
        # === INTERACTION MODULE (Fixed according to paper) ===
        # ===================================================================

        # 3.4.1. Scalar Projection Layer
        # Paper equation (12): I = E_d · E_s
        # This is a batch matrix multiplication: (batch, d, c) @ (batch, c, s) = (batch, d, s)
        interaction_map = torch.bmm(
            drug_encoded,
            se_encoded.transpose(1, 2)
        )  # (batch, d, s)

        # Add channel dimension for CNN: (batch, 1, d, s)
        interaction_map = interaction_map.unsqueeze(1)

        # Apply dropout
        interaction_map = F.dropout(
            interaction_map,
            p=self.dropout.p,
            training=self.training
        )

        # 3.4.2. CNN Layer
        # Paper equation (13): M = CNN(I)
        # Captures local region interactions between neighboring substructures
        interaction_features = self.interaction_cnn(interaction_map)  # (batch, out_channels, d', s')

        # ===================================================================
        # === DECODER (MLP) ===
        # ===================================================================

        # Flatten: M → vector
        interaction_flat = interaction_features.view(batch_size, -1)

        # MLP prediction
        # Paper equations (14-15):
        # O_1 = ReLU(W_1 * Flatten(M) + b_1)
        # Score = W_4 * ReLU(W_3 * ReLU(W_2 * O_1 + b_2) + b_3) + b_4
        score = self.decoder(interaction_flat)  # (batch, 1) raw logits

        return score, drug_encoded, se_encoded

    def get_embeddings(
            self,
            drug: torch.Tensor,
            side_effect: torch.Tensor,
            drug_mask: torch.Tensor,
            se_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return encoder outputs only (no prediction)"""
        with torch.no_grad():
            _, drug_encoded, se_encoded = self.forward(
                drug, side_effect, drug_mask, se_mask
            )
        return drug_encoded, se_encoded

    def count_parameters(self) -> dict:
        """Count trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'drug_encoder': sum(p.numel() for p in self.encoder_drug.parameters()),
            'se_encoder': sum(p.numel() for p in self.encoder_side.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters())
        }


def create_model(config: ModelConfig, device: str = 'cpu') -> DrugSideEffectModel:
    """Factory to create model"""
    model = DrugSideEffectModel(config, device)
    return model.to(device)
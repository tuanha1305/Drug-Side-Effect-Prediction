"""
Main model for drug side effect prediction
Optimized for PyTorch 2.x
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

    Architecture:
        1. Drug Encoder (Transformer)
        2. Side Effect Encoder (Transformer)
        3. Interaction Layer (Outer Product + CNN)
        4. Decoder (MLP)
    """

    def __init__(self, config: ModelConfig, device: str = 'cpu'):
        super(DrugSideEffectModel, self).__init__()

        self.config = config
        self.device = device

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.decoder_dropout)

        # Drug Embedding Layer
        self.emb_drug = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.embedding_dim,
            max_position_size=config.max_position_embeddings,
            dropout_rate=config.dropout_rate
        )

        # Side Effect Embedding Layer
        self.emb_side = Embeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.embedding_dim,
            max_position_size=config.max_position_embeddings,
            dropout_rate=config.dropout_rate
        )

        # Drug Transformer Encoder
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

        # Side Effect Transformer Encoder
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

        # Optional Cross-Attention Encoder
        self.use_cross_attention = config.use_cross_attention
        if self.use_cross_attention:
            self.cross_attention_encoder = Encoder_MultipleLayers(
                n_layer=1,  # Single layer for cross-attention
                hidden_size=config.embedding_dim,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                attention_probs_dropout_prob=config.attention_dropout,
                hidden_dropout_prob=config.hidden_dropout,
                use_flash_attention=config.use_flash_attention,
                use_sdpa=config.use_sdpa
            )

        # Interaction CNN Layer
        self.interaction_cnn = nn.Conv2d(
            in_channels=config.conv_in_channels,
            out_channels=config.conv_out_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_padding
        )

        # Decoder MLP
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
        """
        Build MLP decoder with batch normalization

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization

        Returns:
            Sequential decoder network
        """
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(True))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask in correct format

        Args:
            mask: [batch_size, seq_len] with 1 for valid tokens, 0 for padding

        Returns:
            attention_mask: [batch_size, 1, 1, seq_len] with 0 for valid, -10000 for masked
        """
        # Expand dimensions: [batch, seq_len] -> [batch, 1, 1, seq_len]
        attention_mask = mask.unsqueeze(1).unsqueeze(2)

        # Convert: 1 (valid) -> 0, 0 (masked) -> -10000
        attention_mask = (1.0 - attention_mask) * -10000.0

        return attention_mask

    def forward(
            self,
            drug: torch.Tensor,
            side_effect: torch.Tensor,
            drug_mask: torch.Tensor,
            se_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            drug: Drug indices [batch_size, seq_len]
            side_effect: Side effect indices [batch_size, seq_len]
            drug_mask: Drug attention mask [batch_size, seq_len]
            se_mask: Side effect attention mask [batch_size, seq_len]

        Returns:
            score: Prediction scores [batch_size, 1]
            drug_encoded: Encoded drug features [batch_size, seq_len, hidden_size]
            se_encoded: Encoded side effect features [batch_size, seq_len, hidden_size]
        """
        batch_size = drug.size(0)

        # Move to device
        drug = drug.long().to(self.device)
        side_effect = side_effect.long().to(self.device)
        drug_mask = drug_mask.long().to(self.device)
        se_mask = se_mask.long().to(self.device)

        # Create attention masks
        drug_attention_mask = self._create_attention_mask(drug_mask)
        se_attention_mask = self._create_attention_mask(se_mask)

        # === Drug Encoding ===
        # Embedding: [batch, seq_len, hidden_size]
        drug_emb = self.emb_drug(drug)

        # Encoder: [batch, seq_len, hidden_size]
        drug_encoded = self.encoder_drug(
            drug_emb.float(),
            drug_attention_mask.float(),
            fusion=False
        )

        # === Side Effect Encoding ===
        # Embedding: [batch, seq_len, hidden_size]
        se_emb = self.emb_side(side_effect)

        # Encoder: [batch, seq_len, hidden_size]
        se_encoded = self.encoder_side(
            se_emb.float(),
            se_attention_mask.float(),
            fusion=False
        )

        # === Optional Cross-Attention ===
        if self.use_cross_attention:
            # Stack drug and SE for cross-attention
            combined = torch.cat([drug_encoded, se_encoded], dim=1)
            combined_mask = torch.cat([drug_attention_mask, se_attention_mask], dim=-1)

            # Cross-attention
            combined = self.cross_attention_encoder(
                combined.float(),
                combined_mask.float(),
                fusion=True
            )

            # Split back
            seq_len = drug_encoded.size(1)
            drug_encoded = combined[:, :seq_len, :]
            se_encoded = combined[:, seq_len:, :]

        # === Interaction Layer ===
        # Outer product: [batch, drug_len, 1, hidden] * [batch, 1, se_len, hidden]
        # -> [batch, drug_len, se_len, hidden]
        drug_aug = drug_encoded.unsqueeze(2)  # [batch, drug_len, 1, hidden]
        se_aug = se_encoded.unsqueeze(1)  # [batch, 1, se_len, hidden]

        interaction = drug_aug * se_aug  # [batch, drug_len, se_len, hidden]

        # Permute for CNN: [batch, hidden, drug_len, se_len]
        interaction = interaction.permute(0, 3, 1, 2)

        # Sum over hidden dimension: [batch, drug_len, se_len]
        interaction = torch.sum(interaction, dim=1, keepdim=True)

        # Apply dropout
        interaction = F.dropout(interaction, p=self.dropout.p, training=self.training)

        # Apply CNN: [batch, 1, drug_len, se_len] -> [batch, 3, drug_len-2, se_len-2]
        interaction_features = self.interaction_cnn(interaction)

        # Flatten: [batch, 3 * (drug_len-2) * (se_len-2)]
        interaction_flat = interaction_features.view(batch_size, -1)

        # === Decoder ===
        # MLP: [batch, flattened_dim] -> [batch, 1]
        score = self.decoder(interaction_flat)

        return score, drug_encoded, se_encoded

    def get_embeddings(
            self,
            drug: torch.Tensor,
            side_effect: torch.Tensor,
            drug_mask: torch.Tensor,
            se_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get encoded embeddings without prediction
        Useful for feature extraction

        Returns:
            drug_encoded: [batch, seq_len, hidden_size]
            se_encoded: [batch, seq_len, hidden_size]
        """
        with torch.no_grad():
            _, drug_encoded, se_encoded = self.forward(
                drug, side_effect, drug_mask, se_mask
            )
        return drug_encoded, se_encoded

    def count_parameters(self) -> dict:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count by component
        drug_encoder_params = sum(p.numel() for p in self.encoder_drug.parameters())
        se_encoder_params = sum(p.numel() for p in self.encoder_side.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'drug_encoder': drug_encoder_params,
            'se_encoder': se_encoder_params,
            'decoder': decoder_params
        }


def create_model(config: ModelConfig, device: str = 'cpu') -> DrugSideEffectModel:
    """
    Factory function to create model

    Args:
        config: Model configuration
        device: Device to place model on

    Returns:
        model: DrugSideEffectModel instance
    """
    model = DrugSideEffectModel(config, device)
    model = model.to(device)
    return model


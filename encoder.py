"""
Transformer Encoder components for drug side effect prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """
    Word embedding and position encoding
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_size: int,
        dropout_rate: float
    ):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with optional Flash Attention support
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        use_flash_attention: bool = True,
        use_sdpa: bool = True
    ):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.use_flash_attention = use_flash_attention
        self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape for multi-head attention
        [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fusion: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len] (1 for valid, 0 for masked)
            fusion: whether this is cross-attention (not used in current implementation)
        Returns:
            context_layer: [batch_size, seq_len, hidden_size]
        """
        # Generate Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Convert mask: 1 -> 0 (valid), 0 -> -inf (masked)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            # attention_mask should be [batch_size, 1, 1, seq_len]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

        if self.use_sdpa:
            # Note: mask format is different - None means no masking
            attn_mask = None
            if attention_mask is not None:
                # Convert to boolean mask: True for positions to mask
                attn_mask = (attention_mask < -1).squeeze(1).squeeze(1)
                if not attn_mask.any():
                    attn_mask = None

            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual attention computation
            attention_scores = torch.matmul(
                query_layer,
                key_layer.transpose(-1, -2)
            )
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # Softmax
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply attention to values
            context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfOutput(nn.Module):
    """
    Output of self-attention with residual connection
    """
    def __init__(self, hidden_size: int, hidden_dropout_prob: float):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: output from attention
            input_tensor: residual connection
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    """
    Complete attention layer (self-attention + output)
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        use_flash_attention: bool = True,
        use_sdpa: bool = True
    ):
        super(Attention, self).__init__()
        self.self = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            use_flash_attention,
            use_sdpa
        )
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self,
        input_tensor: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fusion: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_tensor: [batch_size, seq_len, hidden_size]
            attention_mask: attention mask
            fusion: for cross-attention (not used currently)
        """
        self_output = self.self(input_tensor, attention_mask, fusion)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    """
    Intermediate layer (FFN first part)
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    """
    Output layer (FFN second part) with residual connection
    """
    def __init__(
        self,
        intermediate_size: int,
        hidden_size: int,
        hidden_dropout_prob: float
    ):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        use_flash_attention: bool = True,
        use_sdpa: bool = True
    ):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            use_flash_attention,
            use_sdpa
        )
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fusion: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: attention mask
            fusion: for cross-attention
        """
        attention_output = self.attention(hidden_states, attention_mask, fusion)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class EncoderMultipleLayers(nn.Module):
    """
    Multi-layer Transformer encoder
    """
    def __init__(
        self,
        n_layer: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        use_flash_attention: bool = True,
        use_sdpa: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super(EncoderMultipleLayers, self).__init__()

        # Create encoder layers
        self.layer = nn.ModuleList([
            EncoderLayer(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                use_flash_attention,
                use_sdpa
            )
            for _ in range(n_layer)
        ])

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fusion: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
            fusion: for cross-attention
        Returns:
            encoded: [batch_size, seq_len, hidden_size]
        """
        all_encoder_layers = []

        for layer_module in self.layer:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    fusion,
                    use_reentrant=False
                )
            else:
                hidden_states = layer_module(hidden_states, attention_mask, fusion)

            all_encoder_layers.append(hidden_states)

        return hidden_states

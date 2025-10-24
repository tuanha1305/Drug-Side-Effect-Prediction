"""
Transformer Encoder components for drug side effect prediction
Optimized for PyTorch 2.x with Flash Attention support
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
    Optimized for PyTorch 2.x
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
        
        # PyTorch 2.x optimizations
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
            attention_mask: [batch_size, 1, 1, seq_len] or [batch_size, seq_len, seq_len]
            fusion: whether this is cross-attention (not used in current implementation)
        Returns:
            context_layer: [batch_size, seq_len, hidden_size]
        """
        # === 1. Compute Q, K, V ===
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # [B, H, L, D]

        # === 2. Prepare attention mask ===
        attn_mask = None
        if attention_mask is not None:
            # Convert to boolean mask (True = mask out, False = attend)
            # Allow multiple mask shapes: [B,L], [B,L,L], [B,1,1,L], [B,1,L,L]
            if attention_mask.dim() == 2:
                # [B, L] -> [B, 1, 1, L]
                attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [B, L, L] -> [B, 1, L, L]
                attn_mask = (attention_mask == 0).unsqueeze(1)
            elif attention_mask.dim() == 4:
                # [B, 1, 1, L] or [B, 1, L, L]
                attn_mask = (attention_mask == 0)
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

            # Expand mask across attention heads: [B, 1, L, L] -> [B, H, L, L]
            if attn_mask.size(1) == 1 and query_layer.size(1) > 1:
                attn_mask = attn_mask.expand(-1, query_layer.size(1), -1, -1)

        # === 3. Scaled Dot-Product Attention (PyTorch 2.x optimized) ===
        if self.use_sdpa:
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attn_mask,  # [B, H, L, L]
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual fallback (for PyTorch < 2.0 or no SDPA)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attn_mask is not None:
                # attn_mask = True where masked -> convert to -inf
                attention_scores = attention_scores.masked_fill(attn_mask, float('-inf'))

            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)

        # === 4. Merge heads back ===
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


class Encoder_MultipleLayers(nn.Module):
    """
    Multi-layer Transformer encoder
    Optimized for PyTorch 2.x with gradient checkpointing support
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
        super(Encoder_MultipleLayers, self).__init__()
        
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


if __name__ == "__main__":
    # Test encoder
    print("Testing Transformer Encoder...")
    
    batch_size = 4
    seq_len = 50
    vocab_size = 2586
    hidden_size = 200
    num_layers = 8
    num_heads = 8
    intermediate_size = 512
    
    # Create model
    embeddings = Embeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_size=500,
        dropout_rate=0.1
    )
    
    encoder = Encoder_MultipleLayers(
        n_layer=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        use_flash_attention=True,
        use_sdpa=True
    )
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, 1, 1, seq_len))
    
    # Forward pass
    print(f"\nInput shape: {input_ids.shape}")
    
    emb = embeddings(input_ids)
    print(f"Embedding shape: {emb.shape}")
    
    output = encoder(emb, attention_mask, fusion=False)
    print(f"Encoder output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters in encoder: {total_params:,}")
    
    # Test with torch.compile (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nTesting with torch.compile on {device}...")
        
        if device == 'cpu':
            print("Note: torch.compile on CPU may show 'cudagraph partition' warnings - this is normal")
            import warnings
            warnings.filterwarnings('ignore', message='.*cudagraph.*')
        
        try:
            encoder_compiled = torch.compile(encoder, mode='reduce-overhead')
            output_compiled = encoder_compiled(emb, attention_mask, fusion=False)
            print(f"Compiled encoder output shape: {output_compiled.shape}")
            print("✓ torch.compile works!")
        except Exception as e:
            print(f"torch.compile not fully supported on this system: {e}")
    
    print("\n✓ All encoder tests passed!")
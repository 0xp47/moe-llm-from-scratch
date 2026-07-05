"""
AetherTransformer V2 - Modern Small Language Model Architecture
===============================================================

Implements state-of-the-art techniques from 2024-2025 research:
- MobileLLM: Deep & thin architecture, GQA, embedding sharing
- SmolLM: Trapezoidal LR schedule, high-quality data focus  
- TinyStories: Simplified vocabulary for coherent generation
- LLaMA 2/3: RoPE, RMSNorm, SwiGLU, Pre-normalization

Key improvements over V1:
1. Grouped Query Attention (GQA) - reduces KV cache, better efficiency
2. Weight tying (embedding = output head) - fewer params, better perf
3. Deeper/thinner architecture - more layers, smaller hidden dim
4. Proper initialization (small init for output projection)
5. Pre-norm architecture (more stable training)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""
    
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - better extrapolation than learned PE."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # cos, sin: [seq_len, head_dim]
    # q, k: [batch, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation - better than ReLU/GELU for transformers."""
    
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE.
    
    GQA uses fewer key-value heads than query heads, reducing memory
    and compute while maintaining quality. Used in LLaMA 2, Mistral, etc.
    
    - num_heads: number of query heads
    - num_kv_heads: number of key-value heads (must divide num_heads)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        num_kv_heads: int = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # Query projection (full heads)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # Key-Value projection (reduced heads for GQA)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand KV for GQA: repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """
    Modern transformer block with pre-normalization.
    
    Architecture: x -> norm -> attn -> + -> norm -> ffn -> +
    (Pre-norm is more stable than post-norm for training)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        num_kv_heads: int = None,
        ffn_mult: float = 2.67,  # SwiGLU uses 8/3 ≈ 2.67 multiplier
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ) -> None:
        super().__init__()
        
        # Pre-normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Grouped Query Attention
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # SwiGLU FFN (hidden_dim = d_model * ffn_mult, rounded to multiple of 64)
        hidden_dim = int(d_model * ffn_mult)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round up to nearest 64
        self.ffn = SwiGLU(d_model, hidden_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AetherTransformerV2(nn.Module):
    """
    Modern Small Language Model Architecture.
    
    Key design choices from MobileLLM/SmolLM research:
    1. Deep and thin: More layers with smaller d_model
    2. Weight tying: Share embedding and output weights
    3. GQA: Fewer KV heads for efficiency
    4. Small initialization for output projection
    
    Recommended configs:
    - Nano (1M params): d_model=64, num_layers=4, num_heads=4, num_kv_heads=2
    - Tiny (10M params): d_model=256, num_layers=8, num_heads=8, num_kv_heads=4
    - Small (50M params): d_model=512, num_layers=12, num_heads=8, num_kv_heads=4
    """
    
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        num_heads: int = 8,
        num_kv_heads: int = 4,  # GQA: fewer KV heads
        num_layers: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        tie_weights: bool = True,  # Weight tying (recommended)
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm_f = RMSNorm(d_model)
        
        # Output head (optionally tied to embedding)
        if tie_weights:
            self.output_head = None  # Will use embedding weights
        else:
            self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following GPT-2/LLaMA style."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Special scaled init for output projection in residual blocks
        for layer in self.layers:
            # Scale down the output projection of attention and FFN
            torch.nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * len(self.layers)))
            torch.nn.init.normal_(layer.ffn.w3.weight, mean=0.0, std=0.02 / math.sqrt(2 * len(self.layers)))
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token indices
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm_f(x)
        
        # Project to vocabulary
        if self.tie_weights:
            logits = F.linear(x, self.embedding.weight)
        else:
            logits = self.output_head(x)
        
        return logits
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list[int]] = None,
    ) -> Tensor:
        """
        Generate tokens with advanced sampling strategies.
        
        Args:
            input_ids: Starting token sequence [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Keep tokens with cumulative prob < top_p (nucleus sampling)
            repetition_penalty: Penalize repeated tokens (>1 = less repetition)
            stop_tokens: Token IDs that stop generation
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(input_ids)[:, -1, :]  # [batch, vocab]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in set(input_ids[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for stop tokens
            if stop_tokens and next_token.item() in stop_tokens:
                break
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        layer_params = sum(p.numel() for layer in self.layers for p in layer.parameters())
        norm_params = sum(p.numel() for p in self.norm_f.parameters())
        output_params = 0 if self.tie_weights else sum(p.numel() for p in self.output_head.parameters())
        
        total = embedding_params + layer_params + norm_params + output_params
        
        return {
            "embedding": embedding_params,
            "layers": layer_params,
            "norm": norm_params,
            "output_head": output_params,
            "total": total,
            "total_formatted": f"{total / 1e6:.2f}M"
        }


# Convenience function to create model from config
def create_model(config: dict) -> AetherTransformerV2:
    """Create model from config dictionary."""
    return AetherTransformerV2(
        vocab_size=config.get("vocab_size", 8000),
        d_model=config.get("d_model", 256),
        num_heads=config.get("num_heads", 8),
        num_kv_heads=config.get("num_kv_heads", 4),
        num_layers=config.get("num_layers", 8),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_seq_len", 512),
        tie_weights=config.get("tie_weights", True),
    )


# Pre-defined configurations based on research
MODEL_CONFIGS = {
    # ~500K params - for very limited data (<100KB)
    "nano": {
        "d_model": 64,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 6,
        "dropout": 0.1,
    },
    # ~2M params - for small data (100KB-500KB)
    "micro": {
        "d_model": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "num_layers": 8,
        "dropout": 0.1,
    },
    # ~10M params - for medium data (500KB-5MB)
    "tiny": {
        "d_model": 256,
        "num_heads": 8,
        "num_kv_heads": 4,
        "num_layers": 8,
        "dropout": 0.1,
    },
    # ~30M params - for larger data (5MB-50MB)
    "small": {
        "d_model": 384,
        "num_heads": 6,
        "num_kv_heads": 3,
        "num_layers": 12,
        "dropout": 0.1,
    },
}

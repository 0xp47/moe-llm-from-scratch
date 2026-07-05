from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .vision_adapter import VisionAdapter
from .audio_adapter import AudioAdapter
from .moe_layer import MoELayer
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale

class SwiGLU(nn.Module):
    """SwiGLU activation function (Swish-Gated Linear Unit)."""
    
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary position embeddings."""
    
    def __init__(self, d_model: int, nhead: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Transpose for attention: [batch, nhead, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len=seq_len) # v is just for device/dtype
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Flash Attention
        # is_causal=True handles the masking automatically
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(out)

class MultimodalBlock(nn.Module):
    """Transformer block with optional MoE feed-forward layer."""
    
    def __init__(self, d_model: int, nhead: int, is_moe: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if is_moe:
            self.ff: nn.Module = MoELayer(d_model)
        else:
            hidden_dim = int(d_model * 4) 
            self.ff = SwiGLU(d_model, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Attention with residual dropout
        x2 = self.norm1(x)
        attn_out = self.attn(x2)
        x = x + self.dropout(attn_out)
        
        # Feedforward / MoE with residual dropout
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        x = x + self.dropout(ff_out)
        return x

class AetherTransformer(nn.Module):
    """Multimodal Sparse MoE Transformer with optional vision and audio adapters."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1,
        enable_vision: bool = False,
        enable_audio: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio
        
        # Text Embedding with dropout
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Modality Adapters (only if enabled - saves ~86M params!)
        self.vision_adapter = VisionAdapter(embed_dim=d_model) if enable_vision else None
        self.audio_adapter = AudioAdapter(embed_dim=d_model) if enable_audio else None
        
        # NO Absolute Positional Encoding anymore (RoPE handles it)
        
        # Transformer Layers with dropout
        self.layers = nn.ModuleList([
            MultimodalBlock(d_model, nhead, is_moe=(i % 2 == 1), dropout=dropout) 
            for i in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        text_tokens: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        audio: Optional[Tensor] = None
    ) -> Tensor:
        embeddings: list[Tensor] = []
        
        # 1. Process Non-Text Modalities first (only if adapters are enabled)
        if images is not None and self.vision_adapter is not None:
            visual_tokens = self.vision_adapter(images)
            embeddings.append(visual_tokens)
            
        if audio is not None and self.audio_adapter is not None:
            audio_tokens = self.audio_adapter(audio)
            embeddings.append(audio_tokens)

        # 2. Embed Text
        if text_tokens is not None:
            text_embeds = self.text_embedding(text_tokens)
            text_embeds = self.embed_dropout(text_embeds)
            embeddings.append(text_embeds)
            
        if not embeddings:
            raise ValueError("At least one input modality must be provided.")

        # Combine
        x = torch.cat(embeddings, dim=1)
        
        # 3. Pass through Layers (RoPE is applied inside)
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        
        # 4. Output Logits
        logits = self.output_head(x)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: Optional[Tensor] = None,
        images: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0
    ) -> Tensor:
        self.eval()
        generated = text_tokens if text_tokens is not None else torch.empty(1, 0, dtype=torch.long, device=self.text_embedding.weight.device)
        
        for _ in range(max_new_tokens):
            logits = self(text_tokens=generated, images=images, audio=audio)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
        return generated

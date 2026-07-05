"""Tests for AetherAI model components."""
from __future__ import annotations

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer import RMSNorm, SwiGLU, CausalSelfAttention, AetherTransformer
from aether.models.moe_layer import Expert, MoELayer
from aether.models.rope import RotaryEmbedding, rotate_half, apply_rotary_pos_emb


class TestRMSNorm:
    """Tests for RMSNorm normalization layer."""
    
    def test_output_shape(self):
        """RMSNorm should preserve input shape."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization_effect(self):
        """RMSNorm should normalize the input."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64) * 100  # Large values
        out = norm(x)
        # Output should have smaller RMS than input
        rms_in = torch.sqrt(x.pow(2).mean())
        rms_out = torch.sqrt(out.pow(2).mean())
        assert rms_out < rms_in


class TestSwiGLU:
    """Tests for SwiGLU activation."""
    
    def test_output_shape(self):
        """SwiGLU should map d_model -> d_model."""
        swiglu = SwiGLU(d_model=64, hidden_dim=256)
        x = torch.randn(2, 10, 64)
        out = swiglu(x)
        assert out.shape == x.shape
    
    def test_nonlinearity(self):
        """SwiGLU should introduce non-linearity."""
        swiglu = SwiGLU(d_model=32, hidden_dim=128)
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        # f(a + b) != f(a) + f(b) for non-linear functions
        out_sum = swiglu(x1 + x2)
        sum_out = swiglu(x1) + swiglu(x2)
        assert not torch.allclose(out_sum, sum_out, atol=1e-5)


class TestMoELayer:
    """Tests for Mixture-of-Experts layer."""
    
    def test_output_shape(self):
        """MoE layer should preserve input shape."""
        moe = MoELayer(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        out = moe(x)
        assert out.shape == x.shape
    
    def test_expert_count(self):
        """MoE should have correct number of experts."""
        moe = MoELayer(d_model=64, num_experts=8, top_k=2)
        assert len(moe.experts) == 8
    
    def test_sparse_routing(self):
        """Only top-k experts should be activated."""
        moe = MoELayer(d_model=32, num_experts=4, top_k=2)
        x = torch.randn(1, 5, 32)
        
        # Run forward and check gate
        gate_logits = moe.gate(x.view(-1, 32))
        _, selected = torch.topk(torch.softmax(gate_logits, dim=-1), moe.top_k, dim=-1)
        
        # Each token should select exactly top_k experts
        assert selected.shape[-1] == moe.top_k


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings."""
    
    def test_cos_sin_shape(self):
        """RoPE should generate correct cos/sin shapes."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 4, 10, 64)  # [batch, heads, seq, head_dim]
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (1, 1, 10, 64)
        assert sin.shape == (1, 1, 10, 64)
    
    def test_dynamic_extension(self):
        """RoPE should extend for sequences longer than max_position_embeddings."""
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        x = torch.randn(1, 2, 100, 32)  # seq_len > max_position
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (1, 1, 100, 32)


class TestAetherTransformer:
    """Tests for the main transformer model."""
    
    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        return AetherTransformer(vocab_size=100, d_model=64, nhead=2, num_layers=2)
    
    def test_forward_text_only(self, small_model):
        """Model should handle text-only input."""
        tokens = torch.randint(0, 100, (2, 10))
        out = small_model(text_tokens=tokens)
        assert out.shape == (2, 10, 100)  # [batch, seq, vocab]
    
    def test_forward_with_vision(self, small_model):
        """Model should handle text + vision input."""
        tokens = torch.randint(0, 100, (1, 5))
        images = torch.randn(1, 3, 224, 224)
        out = small_model(text_tokens=tokens, images=images)
        # Output seq_len = vision_tokens (64) + text_tokens (5)
        assert out.shape[0] == 1
        assert out.shape[2] == 100
    
    def test_generate(self, small_model):
        """Model should generate new tokens."""
        tokens = torch.randint(0, 100, (1, 5))
        generated = small_model.generate(tokens, max_new_tokens=10)
        assert generated.shape == (1, 15)  # 5 input + 10 generated
    
    def test_no_input_raises(self, small_model):
        """Model should raise error with no input."""
        with pytest.raises(ValueError):
            small_model()


class TestExpert:
    """Tests for individual Expert network."""
    
    def test_expert_output_shape(self):
        """Expert should preserve d_model dimension."""
        expert = Expert(d_model=64, d_ff=256)
        x = torch.randn(10, 64)
        out = expert(x)
        assert out.shape == (10, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

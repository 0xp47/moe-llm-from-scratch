from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Expert(nn.Module):
    """Single expert network using SwiGLU activation."""
    
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # SwiGLU requires 3 matrices. 
        # Traditionally d_ff is larger, but here we keep signature same
        # Hidden dim is typically d_ff / 2 to keep params similar to GELU
        hidden_dim = d_ff 
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU(x) = (SiLU(xW1) * xW2)W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MoELayer(nn.Module):
    """
    Sparsely Gated Mixture-of-Experts Layer.
    Implements Top-K routing with optimized batched expert execution.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.experts = nn.ModuleList([Expert(d_model, d_model * 4) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch * seq_len, d_model]
        num_tokens = x_flat.shape[0]
        
        # Calculate gate scores
        gate_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        weights, selected_experts = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Re-normalize
        
        # Optimized batched routing
        # Pre-allocate output tensor
        out = torch.zeros_like(x_flat)
        
        # Process each expert in parallel-friendly manner
        for expert_idx in range(self.num_experts):
            # Create mask for tokens routed to this expert (across all top-k positions)
            expert_mask = (selected_experts == expert_idx)  # [num_tokens, top_k]
            
            if not expert_mask.any():
                continue
                
            # Get token indices and their corresponding k positions
            token_indices, k_positions = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
            
            # Get unique tokens for this expert (avoid duplicate computation)
            unique_tokens = token_indices.unique()
            
            # Batch process all tokens for this expert at once
            expert_input = x_flat[unique_tokens]  # [num_unique, d_model]
            expert_output = self.experts[expert_idx](expert_input)  # [num_unique, d_model]
            
            # Create mapping from unique to original indices
            unique_to_idx = {t.item(): i for i, t in enumerate(unique_tokens)}
            
            # Accumulate weighted outputs
            for tok_idx, k_pos in zip(token_indices.tolist(), k_positions.tolist()):
                weight = weights[tok_idx, k_pos]
                unique_idx = unique_to_idx[tok_idx]
                out[tok_idx] += weight * expert_output[unique_idx]
                
        return out.view(batch_size, seq_len, d_model)

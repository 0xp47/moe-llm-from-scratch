# Aether-1.0-MoE Architecture Documentation

## Overview

Aether-1.0-MoE is a custom implementation of a Sparse Mixture-of-Experts (MoE) Transformer. It diverges from standard "Dense" models (like GPT-2) by utilizing dynamic routing to specialized expert networks.

## Core Components

### 1. The Sparse MoE Layer

Located in: `src/aether/models/moe_layer.py`

Unlike a standard Feed-Forward Network (FFN) which applies the same weights to every token, our MoE layer consists of:

- **n_experts**: 8 specialized expert networks.
- **Router (Gate)**: A learned linear layer that predicts which experts are best suited for the current token.
- **Top-K Gating**: Only the top 2 experts are activated per token.

**Mathematical Formula:**
$$ y = \sum\_{i \in Top2} G(x)\_i \cdot E_i(x) $$
Where $G(x)$ is the gating probability and $E_i$ is the $i$-th expert.

### 2. SwiGLU Activation

Located in: `src/aether/models/transformer.py` -> `SwiGLU`

We replaced the standard GELU activation with **SwiGLU**, found in Llama 3 and PaLM.
$$ \text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes (xV) $$
This introduces a "gating" mechanism within the neuron itself, allowing for more complex non-linear interactions and "smarter" reasoning capabilities.

### 3. RMSNorm (Root Mean Square Normalization)

Located in: `src/aether/models/transformer.py` -> `RMSNorm`

We utilize RMSNorm instead of LayerNorm. RMSNorm is computationally cheaper (no mean subtraction) and invariant to re-scaling of weights, which improves training stability for deep networks.
$$ \bar{a}\_i = \frac{a_i}{\text{RMS}(a)} g_i, \quad \text{where} \quad \text{RMS}(a) = \sqrt{\frac{1}{n} \sum a_i^2 + \epsilon} $$

## Configuration

Hyperparameters are managed via Hydra in `configs/model/aether_moe.yaml`.

- `d_model`: Embedding dimension (Width)
- `nhead`: Number of attention heads
- `num_layers`: Depth
- `vocab_size`: 441 (Custom BPE)

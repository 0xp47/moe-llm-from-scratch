"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


@pytest.fixture(scope="session")
def device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def small_vocab_size():
    """Small vocabulary for testing."""
    return 100


@pytest.fixture(scope="session")
def small_d_model():
    """Small embedding dimension for testing."""
    return 64


@pytest.fixture
def random_tokens(small_vocab_size):
    """Generate random token sequences."""
    def _generate(batch_size: int = 2, seq_len: int = 10):
        return torch.randint(0, small_vocab_size, (batch_size, seq_len))
    return _generate

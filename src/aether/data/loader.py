"""
Data loading utilities for AetherAI training.

This module provides flexible dataset classes for loading training data from
single files or multiple files, supporting various tokenization schemes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset

if TYPE_CHECKING:
    from typing import Any, Callable, List, Tuple
    from torch import Tensor


class TextDataset(Dataset):
    """
    Dataset for loading text from a single file.
    
    Supports both HuggingFace tokenizers and custom tokenization functions.
    Creates overlapping sequences for language model training where each
    position predicts the next token.
    
    Args:
        file_path: Path to the text file.
        tokenizer: Tokenizer (HuggingFace) or callable for encoding text.
        block_size: Length of each training sequence (context window).
        stride: Step size between sequences. Defaults to block_size (no overlap).
    """
    
    def __init__(
        self, 
        file_path: Union[str, Path], 
        tokenizer: Any, 
        block_size: int = 128,
        stride: Optional[int] = None
    ) -> None:
        self.file_path = Path(file_path)
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        
        # Load and tokenize text
        text = self._load_text()
        self.data = self._tokenize(text, tokenizer)
        
        # Calculate number of valid sequences
        self.n_sequences = max(0, (len(self.data) - self.block_size - 1) // self.stride + 1)
        
        print(f"Loaded {len(self.data):,} tokens from {self.file_path.name} "
              f"({self.n_sequences:,} sequences)")

    def _load_text(self) -> str:
        """Load text from file with proper encoding."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _tokenize(self, text: str, tokenizer: Any) -> Tensor:
        """
        Tokenize text using either HuggingFace tokenizer or callable.
        
        Args:
            text: Raw text to tokenize.
            tokenizer: Tokenizer instance or callable.
            
        Returns:
            Tensor of token IDs.
        """
        # HuggingFace tokenizer
        if hasattr(tokenizer, '__call__') and hasattr(tokenizer, 'encode'):
            return torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # HuggingFace tokenizer with return_tensors
        elif hasattr(tokenizer, '__call__'):
            result = tokenizer(text, return_tensors="pt")
            if hasattr(result, 'input_ids'):
                return result.input_ids.squeeze(0)
            return result.squeeze(0)
        # Simple callable (e.g., character-level)
        else:
            return torch.tensor(tokenizer(text), dtype=torch.long)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Sequence index.
            
        Returns:
            Tuple of (input_ids, target_ids) where target is shifted by 1.
        """
        start = idx * self.stride
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        
        x = chunk[:-1]  # Input: all but last token
        y = chunk[1:]   # Target: all but first token (shifted by 1)
        
        return x, y


class MultiFileDataset(Dataset):
    """
    Dataset that combines multiple text files into one training set.
    
    Useful for training on diverse data sources (conversations, code,
    knowledge bases, etc.) while maintaining a unified interface.
    
    Args:
        file_paths: List of paths to text files.
        tokenizer: Tokenizer for encoding text.
        block_size: Length of each training sequence.
        stride: Step size between sequences.
        separator: String to insert between files (helps model learn boundaries).
    """
    
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        tokenizer: Any,
        block_size: int = 128,
        stride: Optional[int] = None,
        separator: str = "\n\n"
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.separator = separator
        
        # Load all files
        all_text = self._load_all_files(file_paths)
        self.data = self._tokenize(all_text, tokenizer)
        
        # Calculate number of valid sequences
        self.n_sequences = max(0, (len(self.data) - self.block_size - 1) // self.stride + 1)
        
        total_files = len(file_paths)
        print(f"Combined {total_files} files: {len(self.data):,} tokens "
              f"({self.n_sequences:,} sequences)")
    
    def _load_all_files(self, file_paths: List[Union[str, Path]]) -> str:
        """Load and concatenate all files with separator."""
        texts = []
        for path in file_paths:
            path = Path(path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                print(f"  Loaded: {path.name}")
            else:
                print(f"  Warning: {path} not found, skipping")
        return self.separator.join(texts)
    
    def _tokenize(self, text: str, tokenizer: Any) -> Tensor:
        """Tokenize text using provided tokenizer."""
        if hasattr(tokenizer, '__call__') and hasattr(tokenizer, 'encode'):
            return torch.tensor(tokenizer.encode(text), dtype=torch.long)
        elif hasattr(tokenizer, '__call__'):
            result = tokenizer(text, return_tensors="pt")
            if hasattr(result, 'input_ids'):
                return result.input_ids.squeeze(0)
            return result.squeeze(0)
        else:
            return torch.tensor(tokenizer(text), dtype=torch.long)
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = idx * self.stride
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


# Backward compatibility alias
ShakespeareDataset = TextDataset


def create_dataset(
    config: dict,
    tokenizer: Any,
    base_path: Optional[Union[str, Path]] = None
) -> Dataset:
    """
    Factory function to create appropriate dataset from config.
    
    Args:
        config: Configuration dict with 'path' or 'data_files' key.
        tokenizer: Tokenizer for encoding text.
        base_path: Base directory for resolving relative paths.
        
    Returns:
        Dataset instance ready for DataLoader.
    """
    base_path = Path(base_path) if base_path else Path(".")
    block_size = config.get('block_size', 128)
    stride = config.get('stride', None)
    
    # Multiple files configuration
    if 'data_files' in config:
        file_paths = [base_path / f for f in config['data_files']]
        return MultiFileDataset(
            file_paths=file_paths,
            tokenizer=tokenizer,
            block_size=block_size,
            stride=stride
        )
    
    # Single file configuration
    elif 'path' in config:
        return TextDataset(
            file_path=base_path / config['path'],
            tokenizer=tokenizer,
            block_size=block_size,
            stride=stride
        )
    
    else:
        raise ValueError("Config must contain 'path' or 'data_files' key")

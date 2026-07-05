"""
AetherAI Training Script V2 - Modern Training Pipeline
======================================================

Implements best practices from SmolLM, nanoGPT, and TinyStories research:

1. Cosine learning rate schedule with warmup
2. Gradient clipping for stability
3. Mixed precision training (when available)
4. Better logging and checkpointing
5. Validation-based early stopping
6. Weight decay with proper exclusions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast
import os
import sys
import math
import time
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer_v2 import AetherTransformerV2, MODEL_CONFIGS

console = Console(force_terminal=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    """Simple dataset that loads and tokenizes text files."""
    
    def __init__(self, data_files: list[str], tokenizer, block_size: int = 256):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Load and combine all text
        all_text = ""
        for file_path in data_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n"
                console.print(f"  [green]✓[/green] Loaded: {Path(file_path).name}")
        
        # Tokenize
        self.tokens = tokenizer.encode(all_text).ids
        console.print(f"  [cyan]Total tokens: {len(self.tokens):,}[/cyan]")
        
        # Create sequences
        self.num_sequences = (len(self.tokens) - 1) // block_size
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULES
# ═══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine annealing with linear warmup - the gold standard for LLM training."""
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    # Model config
    model_size: str = "micro",
    vocab_size: int = None,
    
    # Data config  
    data_files: list[str] = None,
    tokenizer_path: str = "data/tokenizer.json",
    block_size: int = 256,
    batch_size: int = 32,
    
    # Training config
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
    
    # Early stopping
    patience: int = 15,
    min_delta: float = 0.001,
    
    # System
    device: str = "auto",
    save_dir: str = "checkpoints",
):
    """Main training function with all the modern bells and whistles."""
    
    console.print(Panel.fit(
        "[bold cyan]AetherAI Training V2[/bold cyan]\n"
        "[dim]Modern Small Language Model Training[/dim]",
        border_style="cyan"
    ))
    
    start_time = datetime.now()
    console.print(f"\n[dim]Session: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load tokenizer
    console.print(Panel("[bold]📝 Loading Tokenizer[/bold]", style="cyan", box=box.ROUNDED))
    
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    actual_vocab_size = tokenizer.get_vocab_size()
    console.print(f"  [green]✓[/green] Loaded: {tokenizer_path}")
    console.print(f"  [dim]Vocabulary: {actual_vocab_size:,} tokens[/dim]\n")
    
    if vocab_size is None:
        vocab_size = actual_vocab_size
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]📂 Loading Data[/bold]", style="cyan", box=box.ROUNDED))
    
    if data_files is None:
        # Default: load all .txt files from data/
        data_files = list(Path("data").glob("*.txt"))
    
    dataset = TextDataset(data_files, tokenizer, block_size)
    
    # Train/val split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    console.print(f"  [dim]Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Batches: {len(train_loader):,}[/dim]\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]🧠 Creating Model[/bold]", style="cyan", box=box.ROUNDED))
    
    # Get config
    config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["micro"]).copy()
    config["vocab_size"] = vocab_size
    config["max_seq_len"] = block_size
    
    model = AetherTransformerV2(**config).to(device)
    
    param_info = model.count_parameters()
    
    config_table = Table(box=box.ROUNDED, show_header=False)
    config_table.add_column("", style="cyan")
    config_table.add_column("", style="white")
    config_table.add_row("Model Size", model_size)
    config_table.add_row("Parameters", param_info["total_formatted"])
    config_table.add_row("d_model", str(config["d_model"]))
    config_table.add_row("Layers", str(config["num_layers"]))
    config_table.add_row("Heads (Q/KV)", f"{config['num_heads']}/{config['num_kv_heads']}")
    config_table.add_row("Device", str(device))
    console.print(config_table)
    console.print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────────────────────
    
    # Separate parameters for weight decay (don't decay biases and norms)
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    # Cosine schedule with warmup
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    console.print(f"[dim]Optimizer: AdamW | LR: {learning_rate} | Warmup: {warmup_steps} steps[/dim]\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]🚀 Training[/bold]", style="green", box=box.DOUBLE))
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training", total=len(train_loader), epoch=epoch, total_epochs=epochs)
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                # Forward
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress.update(task, advance=1)
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        console.print(
            f"  [cyan]Epoch {epoch:3d}[/cyan] | "
            f"Train: [yellow]{avg_train_loss:.4f}[/yellow] | "
            f"Val: [{'green' if avg_val_loss < best_val_loss else 'red'}]{avg_val_loss:.4f}[/] | "
            f"LR: {current_lr:.2e}"
        )
        
        # Check for improvement
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            console.print(f"    [green]💾 Saved best model (val_loss: {avg_val_loss:.4f})[/green]")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            console.print(f"\n[yellow]⚠ Early stopping after {patience} epochs without improvement[/yellow]")
            break
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    
    duration = (datetime.now() - start_time).total_seconds()
    
    console.print(Panel(
        f"[bold green]✨ Training Complete![/bold green]\n\n"
        f"Best Val Loss: [cyan]{best_val_loss:.4f}[/cyan]\n"
        f"Total Time: [cyan]{duration/60:.1f} minutes[/cyan]\n"
        f"Model saved to: [cyan]{save_dir}/best_model.pt[/cyan]",
        title="Summary",
        border_style="green"
    ))
    
    return model, best_val_loss


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AetherAI Training V2")
    
    # Model
    parser.add_argument("--model-size", type=str, default="micro", choices=["nano", "micro", "tiny", "small"])
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    
    # System
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    # Find data files
    data_files = list(Path(args.data_dir).glob("*.txt"))
    
    train(
        model_size=args.model_size,
        data_files=data_files,
        tokenizer_path=args.tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir,
    )

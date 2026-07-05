"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        AETHER AI - NEURAL TRAINING ENGINE                     ║
║                     Advanced Transformer Training Pipeline                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer import AetherTransformer
from aether.data.loader import create_dataset

# Initialize Rich console
console = Console(force_terminal=True)
import logging
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLING & THEMING
# ═══════════════════════════════════════════════════════════════════════════════

def create_header():
    """Create a stylish ASCII header."""
    header = """
[bold cyan]    ╔═══════════════════════════════════════════════════════════════════════╗
    ║[/bold cyan] [bold white]█████╗ ███████╗████████╗██╗  ██╗███████╗██████╗     █████╗ ██╗[/bold white] [bold cyan]║
    ║[/bold cyan] [bold white]██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗   ██╔══██╗██║[/bold white] [bold cyan]║
    ║[/bold cyan] [bold white]███████║█████╗     ██║   ███████║█████╗  ██████╔╝   ███████║██║[/bold white] [bold cyan]║
    ║[/bold cyan] [bold white]██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗   ██╔══██║██║[/bold white] [bold cyan]║
    ║[/bold cyan] [bold white]██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║   ██║  ██║██║[/bold white] [bold cyan]║
    ║[/bold cyan] [bold white]╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝[/bold white] [bold cyan]║
    ╠═══════════════════════════════════════════════════════════════════════╣
    ║[/bold cyan]        [dim]Neural Training Engine v2.5[/dim]  •  [dim]Sparse MoE Transformer[/dim]        [bold cyan]║
    ╚═══════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""
    return header

def create_config_panel(cfg: DictConfig) -> Panel:
    """Create a beautiful configuration display panel."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)
    table.add_column("Parameter", style="cyan", width=20)
    table.add_column("Value", style="white", width=15)
    
    table.add_row(
        "🧠 Model Dim", str(cfg.model.d_model),
        "🔢 Vocab Size", str(cfg.model.vocab_size)
    )
    table.add_row(
        "👁️ Attention Heads", str(cfg.model.nhead),
        "📚 Layers", str(cfg.model.num_layers)
    )
    table.add_row(
        "📦 Batch Size", str(cfg.data.batch_size),
        "📏 Block Size", str(cfg.data.block_size)
    )
    table.add_row(
        "🎯 Learning Rate", str(cfg.train.lr),
        "🔄 Epochs", str(cfg.train.epochs)
    )
    
    return Panel(
        table,
        title="[bold white]⚙️  Configuration[/bold white]",
        border_style="cyan",
        box=box.DOUBLE
    )

def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def count_parameters(model: nn.Module) -> str:
    """Count and format model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e9:
        return f"{total/1e9:.2f}B ({trainable/1e9:.2f}B trainable)"
    elif total >= 1e6:
        return f"{total/1e6:.2f}M ({trainable/1e6:.2f}M trainable)"
    else:
        return f"{total/1e3:.2f}K ({trainable/1e3:.2f}K trainable)"

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(text_tokens=x, images=None, audio=None)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss = criterion(logits, y)
            total_loss += loss.item()
    if len(val_loader) == 0:
        return 0.0
    return total_loss / len(val_loader)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_pro(cfg: DictConfig):
    """Main training function with beautiful UI."""
    
    console.clear()
    console.print(create_header())
    
    start_time = datetime.now()
    console.print(f"[dim]Session started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
    
    log.info(f"Training started with config:\n{OmegaConf.to_yaml(cfg)}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    with console.status("[bold cyan]Initializing training environment...[/bold cyan]", spinner="dots12"):
        device_str = cfg.train.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            device_info = f"{gpu_name} ({gpu_mem:.1f}GB)"
        else:
            device_info = "CPU Mode"
        
        os.makedirs(cfg.train.save_dir, exist_ok=True)
        time.sleep(0.3)
    
    console.print(create_config_panel(cfg))
    console.print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: TOKENIZER
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]📝 Loading Tokenizer[/bold]", style="cyan", box=box.ROUNDED))
    
    custom_tok_path = "data/tokenizer.json"
    if os.path.exists(custom_tok_path):
        try:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tok_path)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            console.print(f"   [green]✓[/green] Custom tokenizer loaded: [cyan]{custom_tok_path}[/cyan]")
            console.print(f"   [dim]  Vocabulary size: {len(tokenizer):,} tokens[/dim]")
            log.info(f"Loading custom tokenizer: {custom_tok_path}")
        except Exception as e:
            console.print(f"   [yellow]⚠[/yellow] Custom tokenizer failed, using GPT-2 fallback")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        console.print(f"   [green]✓[/green] GPT-2 tokenizer loaded")
    
    console.print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]📂 Loading Training Data[/bold]", style="cyan", box=box.ROUNDED))
    
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    full_dataset = create_dataset(data_config, tokenizer, base_path=".")
    
    if len(full_dataset) < 10:
        console.print(f"   [yellow]⚠[/yellow] Dataset small: {len(full_dataset)} sequences")
    
    train_len = int(cfg.data.train_split * len(full_dataset))
    val_len = len(full_dataset) - train_len
    if train_len == 0:
        train_len = len(full_dataset)
    if val_len == 0:
        val_len = 0
    
    if val_len > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
        val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.system.num_workers)
    else:
        train_dataset = full_dataset
        val_loader = []

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.system.num_workers)
    
    data_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    data_table.add_column("", style="dim")
    data_table.add_column("", style="white")
    data_table.add_row("   Training samples", f"[green]{len(train_dataset):,}[/green]")
    data_table.add_row("   Validation samples", f"[blue]{val_len:,}[/blue]")
    data_table.add_row("   Batches per epoch", f"[cyan]{len(train_loader):,}[/cyan]")
    console.print(data_table)
    console.print()
    
    log.info(f"Data Loaded: {len(train_dataset)} samples.")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: MODEL INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]🧠 Initializing Neural Network[/bold]", style="cyan", box=box.ROUNDED))
    
    dropout = getattr(cfg.model, 'dropout', 0.1)
    enable_vision = getattr(cfg.model, 'enable_vision', False)
    enable_audio = getattr(cfg.model, 'enable_audio', False)
    model = AetherTransformer(
        vocab_size=len(tokenizer),
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=dropout,
        enable_vision=enable_vision,
        enable_audio=enable_audio
    ).to(device)
    
    # Use weight decay for regularization
    weight_decay = getattr(cfg.train, 'weight_decay', 0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping configuration
    patience = getattr(cfg.train, 'patience', 5)  # Stop if no improvement for N epochs
    min_delta = getattr(cfg.train, 'min_delta', 0.01)  # Minimum improvement required
    
    model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    model_table.add_column("", style="dim")
    model_table.add_column("", style="white")
    model_table.add_row("   Architecture", "[cyan]AetherTransformer (Sparse MoE)[/cyan]")
    model_table.add_row("   Parameters", f"[green]{count_parameters(model)}[/green]")
    model_table.add_row("   Device", f"[yellow]{device_info}[/yellow]")
    model_table.add_row("   Optimizer", "[white]AdamW[/white]")
    console.print(model_table)
    console.print()
    
    log.info(f"Model Initialized. Vocab size: {len(tokenizer)}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    
    console.print(Panel("[bold]🚀 Starting Training[/bold]", style="green", box=box.DOUBLE))
    console.print()
    
    best_val_loss = float('inf')
    epochs = cfg.train.epochs
    training_start = time.time()
    history = {"train_loss": [], "val_loss": []}
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0
        
        epoch_text = Text()
        epoch_text.append("━" * 20 + " ", style="dim cyan")
        epoch_text.append(f"EPOCH {epoch+1}/{epochs}", style="bold white")
        epoch_text.append(" " + "━" * 20, style="dim cyan")
        console.print(epoch_text)
        
        log.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        with Progress(
            SpinnerColumn(spinner_name="dots12", style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            
            task = progress.add_task("Training", total=len(train_loader))
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = model(text_tokens=x)
                
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), y.view(B*T))
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                progress.update(task, advance=1, description=f"Training [dim](loss: {loss.item():.4f})[/dim]")
                
                if batch_idx % 20 == 0:
                    log.info(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss = 0.0
        
        if val_loader:
            with console.status("[cyan]Evaluating on validation set...[/cyan]", spinner="dots"):
                val_loss = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        
        summary_table = Table(box=box.ROUNDED, show_header=False, border_style="dim")
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", justify="right")
        
        loss_style = "green" if val_loss < best_val_loss else "white"
        
        summary_table.add_row("📉 Train Loss", f"[cyan]{avg_train_loss:.4f}[/cyan]")
        summary_table.add_row("📊 Val Loss", f"[{loss_style}]{val_loss:.4f}[/{loss_style}]")
        summary_table.add_row("⏱️  Epoch Time", f"[yellow]{format_time(epoch_time)}[/yellow]")
        
        console.print(summary_table)
        
        log.info(f"Epoch {epoch+1} Complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loader and val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_path = os.path.join(cfg.train.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            console.print(f"   [green]💾 New best model saved![/green] [dim](val_loss: {val_loss:.4f})[/dim]")
            log.info("New best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                console.print(f"\n   [yellow]⚠ Early stopping triggered![/yellow] No improvement for {patience} epochs.")
                log.info(f"Early stopping: No improvement for {patience} epochs.")
                break
        
        torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, "last_model.pt"))
        console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: TRAINING COMPLETE
    # ─────────────────────────────────────────────────────────────────────────
    
    total_time = time.time() - training_start
    
    final_panel = Panel(
        f"""
[bold green]✨ Training Complete![/bold green]

[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]

   📊 [cyan]Final Train Loss:[/cyan]  {history['train_loss'][-1]:.4f}
   📈 [cyan]Final Val Loss:[/cyan]    {history['val_loss'][-1] if history['val_loss'][-1] else 'N/A':.4f}
   🏆 [cyan]Best Val Loss:[/cyan]     {best_val_loss:.4f}
   
   ⏱️  [yellow]Total Time:[/yellow]        {format_time(total_time)}
   💾 [yellow]Model Saved:[/yellow]       {cfg.train.save_dir}/best_model.pt

[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
        """,
        title="[bold white]🎯 Training Summary[/bold white]",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(final_panel)
    
    torch.save(model.state_dict(), os.path.join(cfg.train.save_dir, "best_model.pt"))
    log.info("Training Complete. Model saved.")
    
    end_time = datetime.now()
    console.print(f"\n[dim]Session ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

if __name__ == "__main__":
    train_pro()

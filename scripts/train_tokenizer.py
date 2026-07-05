"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    AETHER AI - TOKENIZER TRAINING ENGINE                      ║
║                      BPE Tokenizer for Language Models                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os
import glob
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Train on ALL data files (including subfolders)
DATA_DIR = "data"
DATA_FILES = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
DATA_FILES = list(set(DATA_FILES))  # Remove duplicates
OUTPUT_FILE = "data/tokenizer.json"

def train_tokenizer():
    console.print(Panel(
        "[bold cyan]AETHER TOKENIZER TRAINER[/bold cyan]\n"
        "[dim]Building vocabulary from training data[/dim]",
        box=box.DOUBLE
    ))
    
    # Show files being used
    table = Table(title="📂 Training Files", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green", justify="right")
    
    total_chars = 0
    valid_files = []
    
    for f in DATA_FILES:
        if os.path.exists(f):
            size = os.path.getsize(f)
            total_chars += size
            valid_files.append(f)
            table.add_row(os.path.basename(f), f"{size/1024:.1f} KB")
    
    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(valid_files)} files, {total_chars/1024:.1f} KB\n")
    
    if not valid_files:
        console.print("[red]❌ No data files found![/red]")
        return
    
    # Initialize tokenizer with ByteLevel BPE (like GPT-2)
    console.print("[cyan]⚙️  Initializing BPE tokenizer...[/cyan]")
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Normalizer: Unicode normalization
    tokenizer.normalizer = NFKC()
    
    # Pre-tokenizer: ByteLevel (handles all unicode, like GPT-2)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    # Decoder
    tokenizer.decoder = ByteLevelDecoder()

    # Trainer with larger vocab for better coverage
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]", "[MASK]", "[CLS]"],
        vocab_size=8000,           # Larger vocab for better coverage
        min_frequency=2,           # Include tokens appearing at least twice
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    # Train on all files
    console.print("[cyan]🚀 Training tokenizer...[/cyan]")
    tokenizer.train(valid_files, trainer)
    
    # Save
    tokenizer.save(OUTPUT_FILE)
    
    # Summary
    console.print(Panel(
        f"[bold green]✅ Tokenizer Training Complete![/bold green]\n\n"
        f"   📊 Vocabulary Size: [cyan]{tokenizer.get_vocab_size():,}[/cyan] tokens\n"
        f"   💾 Saved to: [yellow]{OUTPUT_FILE}[/yellow]\n"
        f"   📁 Trained on: [white]{len(valid_files)} files[/white]",
        title="🎯 Results",
        border_style="green",
        box=box.DOUBLE
    ))
    
    # Test encoding
    console.print("\n[dim]Testing tokenizer...[/dim]")
    test_texts = [
        "Hello, I am Aether!",
        "What is machine learning?",
        "User: Hi\nSystem: Hello!"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        console.print(f"  [dim]{text[:30]}...[/dim] → [cyan]{len(encoded.ids)} tokens[/cyan]")

if __name__ == "__main__":
    train_tokenizer()

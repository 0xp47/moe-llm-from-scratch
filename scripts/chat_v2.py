"""
AetherAI Chat V2 - Interactive Chat with Modern Model
=====================================================
"""

import torch
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer_v2 import AetherTransformerV2

console = Console()


def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Load model from checkpoint."""
    from tokenizers import Tokenizer
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    console.print(f"[green]✓[/green] Loaded tokenizer ({tokenizer.get_vocab_size()} tokens)")
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    console.print(f"[green]✓[/green] Loaded checkpoint: {checkpoint_path}")
    console.print(f"[dim]  Config: d_model={config['d_model']}, layers={config['num_layers']}[/dim]")
    
    # Create and load model
    model = AetherTransformerV2(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓[/green] Model loaded ({param_count/1e6:.2f}M parameters)")
    
    return model, tokenizer, device


def generate_response(
    model, 
    tokenizer, 
    device,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Generate a response to the prompt."""
    # Format prompt
    formatted = f"User: {prompt}\nSystem:"
    
    # Tokenize
    input_ids = tokenizer.encode(formatted).ids
    input_tensor = torch.tensor([input_ids], device=device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.15,
        )
    
    # Decode
    generated_ids = output[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    
    # Extract response (after "System:")
    if "System:" in full_text:
        response = full_text.split("System:")[-1].strip()
        # Stop at next "User:" if present
        if "User:" in response:
            response = response.split("User:")[0].strip()
        return response
    
    return full_text


def main():
    console.print(Panel.fit(
        "[bold cyan]AetherAI Chat V2[/bold cyan]\n"
        "[dim]Modern Small Language Model[/dim]",
        border_style="cyan"
    ))
    
    # Load model
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        console.print("[yellow]Make sure to train the model first with train_v2.py[/yellow]")
        return
    
    console.print("\n[dim]Type 'quit' or 'exit' to end the conversation.[/dim]\n")
    
    # Chat loop
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("[cyan]Goodbye![/cyan]")
                break
            
            if not user_input.strip():
                continue
            
            # Generate response
            response = generate_response(model, tokenizer, device, user_input)
            
            console.print(Panel(
                response,
                title="[bold cyan]Aether[/bold cyan]",
                border_style="cyan",
            ))
            
        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

import torch
import time
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from transformers import AutoTokenizer

import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer import AetherTransformer
from aether.core.safety import SafetyGuardrail
from aether.core.tools import registry
from aether.core.nlu import IntentClassifier

# --- Configuration ---
console = Console()
nlu_engine = IntentClassifier()

def startup_sequence():
    """Simulates a professional boot sequence."""
    console.clear()
    
    with console.status("[bold cyan]Initializing Project Aether Core Systems...", spinner="dots"):
        time.sleep(1)
        console.log("[green]✓[/green] Memory Bank Loaded")
        time.sleep(0.3)
        console.log("[green]✓[/green] Neural Engine Online (CUDA/CPU)")
        time.sleep(0.3)
        console.log("[green]✓[/green] Safety Protocols Active")
        time.sleep(0.3)
        console.log("[green]✓[/green] Tool Registry Mounted")
        time.sleep(0.5) 
    
    console.print(Panel.fit(
        "[bold white]PROJECT AETHER[/bold white]\n[cyan]Advanced Multimodal Intelligence[/cyan]\nv2.4.0-Enterprise",
        style="bold blue",
        box=box.DOUBLE
    ))
    console.print("[dim]Type 'exit' to disconnect.[/dim]\n")

def render_agent_response(text, title="Aether", style="green", meta=None):
    """Renders the agent's response in a panel."""
    content = text
    if meta:
        content += f"\n\n[dim italic]Meta: {meta}[/dim italic]"
    console.print(Panel(content, title=title, style=style, border_style=style))

def render_system_log(intent, confidence, tool_used=None):
    """Renders a small log table for transparency."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_row("[bold cyan]Intent:[/bold cyan]", intent)
    table.add_row("[bold cyan]Confidence:[/bold cyan]", f"{confidence:.2f}")
    if tool_used:
        table.add_row("[bold yellow]Tool Call:[/bold yellow]", tool_used)
    
    console.print(table)

def main():
    startup_sequence()

    # 1. Load Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    
    # Check for custom tokenizer
    custom_tok_path = "data/tokenizer.json"
    if os.path.exists(custom_tok_path):
        console.log(f"[bold yellow]Loading Custom Tokenizer from {custom_tok_path}[/bold yellow]")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tok_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load checkpoint and detect model configuration
    checkpoint_path = "checkpoints/best_model.pt"
    d_model, nhead, num_layers, dropout = 256, 4, 4, 0.1  # defaults
    
    if os.path.exists(checkpoint_path):
        console.log(f"[green]Found Checkpoint:[/green] {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Detect model dimensions from checkpoint
        if "text_embedding.weight" in state_dict:
            vocab_size, d_model = state_dict["text_embedding.weight"].shape
            console.log(f"[cyan]Detected model: d_model={d_model}, vocab_size={vocab_size}[/cyan]")
        
        # Count layers
        layer_count = 0
        for key in state_dict.keys():
            if key.startswith("layers.") and ".norm1.scale" in key:
                layer_count += 1
        if layer_count > 0:
            num_layers = layer_count
            console.log(f"[cyan]Detected layers: {num_layers}[/cyan]")
        
        # Initialize model with detected config
        model = AetherTransformer(vocab_size=len(tokenizer), d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
        model.load_state_dict(state_dict)
    else:
        console.log("[yellow]No checkpoint found. Using random weights.[/yellow]")
        model = AetherTransformer(vocab_size=len(tokenizer), d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
    
    guardrail = SafetyGuardrail()

    # 2. Main Loop
    while True:
        try:
            # User Input
            user_input = console.input("\n[bold white]User > [/bold white]")
            
            # --- Analysis Phase ---
            with console.status("[bold yellow]Analyzing intent...", spinner="flip"):
                time.sleep(0.4) # Simulate processing
                intent, confidence, meta = nlu_engine.classify(user_input)

            if intent == "SYSTEM_CONTROL":
                console.print("[red]System shutdown sequence initiated...[/red]")
                break

            # --- Safety Phase ---
            is_safe, reason = guardrail.validate_input(user_input)
            if not is_safe:
                render_system_log(intent, confidence)
                console.print(Panel(
                    f"[bold red]SECURITY ALERT[/bold red]\n{reason}",
                    style="red",
                    title="Guardrail Interception"
                ))
                console.print(f"[dim]Refusal Sent: {guardrail.format_refusal()}[/dim]")
                continue

            # --- Execution Phase ---
            response_text = ""
            style = "green"
            tool_info = None

            if intent == "CALCULATION":
                with console.status("[bold blue]Running calculation engine...", spinner="arc"):
                    time.sleep(0.5)
                    result = registry.execute("calculator", meta["expression"])
                    response_text = f"The result of your calculation is: [bold]{result}[/bold]"
                    style = "blue"
                    tool_info = f"Calculator('{meta['expression']}')"

            elif intent == "WEATHER":
                with console.status(f"[bold blue]Connecting to weather satellite ({meta['location']})...", spinner="earth"):
                    time.sleep(0.8)
                    result = registry.execute("get_weather", meta["location"])
                    response_text = f"Current conditions in {meta['location']}: [bold]{result}[/bold]"
                    style = "cyan"
                    tool_info = f"WeatherAPI('{meta['location']}')"
            
            else: # GENERAL_CHAT
                with console.status("[bold green]Neural generation in progress...", spinner="aesthetic"):
                    # Neural Inference - text only
                    # Format prompt to match training data (User: ...\nSystem: ...)
                    prompt = f"User: {user_input}\nSystem:"
                    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        gen_ids = model.generate(
                            text_tokens=inputs,
                            max_new_tokens=60, 
                            temperature=0.8
                        )
                    
                    full_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    
                    # Extract only the System's response (everything after the prompt)
                    if prompt in full_output:
                        response_text = full_output.split(prompt)[-1].strip()
                        # Stop at next User prompt if generated
                        if "User:" in response_text:
                            response_text = response_text.split("User:")[0].strip()
                    else:
                        response_text = full_output
                        
                    # Output Guardrail
                    is_out_safe, out_reason = guardrail.validate_output(response_text)
                    
                    if not is_out_safe:
                        response_text = "[REDACTED] - Output violated safety policy."
                        style = "red"
            
            # --- Render Output ---
            render_system_log(intent, confidence, tool_used=tool_info)
            render_agent_response(response_text, style=style)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

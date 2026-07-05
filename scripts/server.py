from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aether.models.transformer import AetherTransformer
from aether.core.agent import AetherAgent
from aether.core.safety import SafetyGuardrail
from aether.core.nlu import IntentClassifier
from aether.core.tools import registry


# --- Global State ---
class AppState:
    model: AetherTransformer | None = None
    tokenizer: Any = None
    device: torch.device | None = None
    agent: AetherAgent | None = None
    guardrail: SafetyGuardrail | None = None
    nlu: IntentClassifier | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler (replaces deprecated on_event)."""
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    
    print("🚀 Initializing AetherAI Model...")
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    custom_tok_path = "data/tokenizer.json"
    if os.path.exists(custom_tok_path):
        print(f"📚 Loading Custom Tokenizer from {custom_tok_path}")
        state.tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tok_path)
    else:
        state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if state.tokenizer.pad_token is None:
        state.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load model
    state.model = AetherTransformer(
        vocab_size=len(state.tokenizer), 
        d_model=256, 
        nhead=4, 
        num_layers=4
    ).to(state.device)
    
    checkpoint_path = os.path.join("checkpoints", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading weights from {checkpoint_path}")
        state.model.load_state_dict(torch.load(checkpoint_path, map_location=state.device))
    else:
        print("⚠️  No checkpoint found! Model is random.")
    
    # Initialize agent and safety systems
    state.agent = AetherAgent(state.model, state.tokenizer, state.device)
    state.guardrail = SafetyGuardrail()
    state.nlu = IntentClassifier()
    
    print("🌟 AetherAI is ONLINE.")
    
    yield  # Server runs here
    
    # Cleanup
    print("👋 Shutting down AetherAI...")


app = FastAPI(
    title="AetherAI API", 
    version="2.5.0", 
    description="Enterprise Multimodal LLM API with ReAct Agent",
    lifespan=lifespan
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    max_tokens: int = 50
    temperature: float = 0.7
    use_agent: bool = False  # Enable ReAct agent mode


class AgentRequest(BaseModel):
    prompt: str
    max_steps: int = 3

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    user_msg = request.messages[-1]['content']
    
    # Safety check
    is_safe, reason = state.guardrail.validate_input(user_msg)
    if not is_safe:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": state.guardrail.format_refusal()
                },
                "finish_reason": "content_filter"
            }]
        }
    
    # Use agent mode if requested
    if request.use_agent:
        response = state.agent.run(user_msg)
        output_text = response
    else:
        # Standard inference
        inputs = state.tokenizer(user_msg, return_tensors="pt").input_ids.to(state.device)
        
        gen_ids = state.model.generate(
            text_tokens=inputs, 
            max_new_tokens=request.max_tokens, 
            temperature=request.temperature
        )
        
        output_text = state.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if output_text.startswith(user_msg):
            output_text = output_text[len(user_msg):]
    
    # Post-generation safety check
    is_out_safe, _ = state.guardrail.validate_output(output_text)
    if not is_out_safe:
        output_text = "[Response filtered for safety]"

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "finish_reason": "stop"
        }]
    }


@app.post("/v1/agent/run")
async def run_agent(request: AgentRequest):
    """Run the ReAct agent with tool access."""
    # Safety check
    is_safe, reason = state.guardrail.validate_input(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=reason)
    
    state.agent.max_steps = request.max_steps
    response = state.agent.run(request.prompt)
    
    return {
        "response": response,
        "tools_available": list(registry.tools.keys())
    }


@app.get("/v1/tools")
def list_tools():
    """List all available tools for the agent."""
    return {
        "tools": [
            {
                "name": name,
                "description": data["schema"]["description"]
            }
            for name, data in registry.tools.items()
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "device": str(state.device),
        "model_loaded": state.model is not None,
        "agent_ready": state.agent is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

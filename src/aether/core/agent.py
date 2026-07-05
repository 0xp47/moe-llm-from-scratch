from __future__ import annotations

import torch
import json
import re
from typing import TYPE_CHECKING, Any

from .tools import registry

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedTokenizerBase
    from aether.models.transformer import AetherTransformer


class AetherAgent:
    """
    Implements a ReAct (Reasoning + Acting) Loop for 'Gemini-like' agentic behavior.
    """
    
    def __init__(
        self,
        model: AetherTransformer,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = 3

    def run(self, user_prompt: str) -> str:
        """
        Executes the ReAct loop: Thought -> Action -> Observation -> Final Answer.
        """
        context = f"""System: You are Aether, an advanced AI Assistant designed by 0x3ef8.
Instructions:
1. You must "think" before you act.
2. Use the format:
Thought: [Your internal reasoning]
Action: [Tool Name]
Action Input: [Tool Arguments]
Observation: [Tool Output]
3. If no tool is needed, just answer directly.

User: {user_prompt}
"""
        print(f"--- [Agent Started] ---")
        
        for step in range(self.max_steps):
            # 1. Model Generation (Simulated for pre-trained state)
            # In production: response = self.model.generate(context)
            response = self._simulate_inference(user_prompt, step, context)
            
            print(f"\n[Step {step+1}] Model Output:\n{response}")
            
            # 2. Parse Action
            action_match = re.search(r"Action: (\w+)", response)
            input_match = re.search(r"Action Input: (.*)", response)
            
            if action_match and input_match:
                tool_name = action_match.group(1)
                tool_args = input_match.group(1)
                
                print(f"👉 Tool Detected: {tool_name}('{tool_args}')")
                
                # 3. Execution
                observation = registry.execute(tool_name.strip(), tool_args.strip())
                print(f"👀 Observation: {observation}")
                
                # 4. Update Context
                context += f"\n{response}\nObservation: {observation}\n"
                
                # If we have an observation, let the model synthesize the final answer next
                # For this demo, we force a final answer if we got a result
                return f"Final Answer: {observation}"
            
            elif "Final Answer:" in response:
                return response.split("Final Answer:")[1].strip()
                
        return "I could not complete the request within the step limit."

    def _simulate_inference(self, prompt: str, step: int, current_context: str) -> str:
        """
        Simulates the ReAct process because the base model isn't fine-tuned for it yet.
        """
        prompt_lower = prompt.lower()
        
        # Scenario: Math
        if ("calculate" in prompt_lower or "+" in prompt_lower) and step == 0:
            return """Thought: The user wants to perform a calculation. I should use the calculator tool.
Action: calculator
Action Input: 123 * 45"""
            
        # Scenario: Weather
        if "weather" in prompt_lower and step == 0:
            loc = "London"
            if "tokyo" in prompt_lower: loc = "Tokyo"
            return f"""Thought: The user is asking for weather information. I need to check the weather API.
Action: get_weather
Action Input: {loc}"""

        # Final Answer fallback
        return "Thought: I have sufficient information.\nFinal Answer: I can help you with that directly."

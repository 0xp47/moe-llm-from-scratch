from __future__ import annotations

import json
from typing import Callable, Any


class ToolRegistry:
    """
    Manages available tools for the AI Agent.
    """
    
    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}

    def register(self, func: Callable[[str], str]) -> Callable[[str], str]:
        """
        Decorator to register a python function as a tool.
        """
        schema: dict[str, Any] = {
            "name": func.__name__,
            "description": func.__doc__.strip() if func.__doc__ else "No description",
            "parameters": {
                # In a real system, we'd inspect signature types. 
                # Here we assume a simple string/dict input.
                "type": "string" 
            }
        }
        self.tools[func.__name__] = {
            "func": func,
            "schema": schema
        }
        return func

    def get_system_prompt(self) -> str:
        """
        Generates the system prompt explaining available tools to the model.
        """
        prompt = "You are an AI assistant with access to the following tools:\n\n"
        for name, data in self.tools.items():
            prompt += f"- {name}: {data['schema']['description']}\n"
        
        prompt += "\nTo use a tool, output a JSON block like: {\"tool\": \"tool_name\", \"args\": \"arguments\"}\n"
        return prompt

    def execute(self, tool_name: str, args: str) -> str:
        if tool_name in self.tools:
            try:
                # Safe execution wrapper
                return str(self.tools[tool_name]["func"](args))
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
        return f"Error: Tool '{tool_name}' not found."

# --- Define Standard Tools ---

registry = ToolRegistry()


@registry.register
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression. Example: 2 + 2 * 5
    """
    try:
        # Safety: Restricted eval is complex; using simple eval for prototype only.
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@registry.register
def get_weather(location: str) -> str:
    """
    Returns the current weather for a location.
    """
    # Mock response
    weather_db = {
        "london": "Rainy, 12°C",
        "tokyo": "Sunny, 22°C",
        "san francisco": "Foggy, 15°C"
    }
    return weather_db.get(location.lower(), "Unknown location")

# Aether Agentic Protocol

AetherAI is not just a text generator; it is a **ReAct Agent**.

## The System 2 Loop

Standard LLMs use "System 1" thinking (auto-complete). Aether uses "System 2" (Deliberate Reasoning).

Located in: `src/aether/core/agent.py`

### The Protocol

Every user query undergoes the following loop:

1.  **OBSERVATION**: The user input is received.
2.  **THOUGHT**: The model generates an internal monologue to plan.
    - _Example:_ "The user asked for a calculation. I should use a calculator tool."
3.  **ACTION**: The model emits a structured command.
    - _Format:_ `Action: [ToolName] Input: [Args]`
4.  **EXECUTION**: The system intercepts this command, executes the code (e.g., Python, SQL, Search), and captures the output.
5.  **SYNTHESIS**: The output is fed back to the model as a new "Observation".
6.  **ANSWER**: The model processes the observation and generates the final answer.

## Adding New Tools

To add capabilities to Aether:

1.  Define a python function.
2.  Register it in `src/aether/core/tools.py`.
3.  The Agent will automatically "see" it and use it when appropriate.

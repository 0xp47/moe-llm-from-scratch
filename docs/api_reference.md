# AetherAI API Reference

The AetherAI Server (`scripts/server.py`) implements an **OpenAI-Compatible** REST API. This allows it to be used as a drop-in replacement for GPT-4 in many tools.

## Base URL

`http://localhost:8000`

## Endpoints

### 1. Chat Completions

**POST** `/v1/chat/completions`

Generates a response from the model based on a conversation history.

**Request Body:**

```json
{
  "model": "aether-1.0-moe",
  "messages": [{ "role": "user", "content": "Hello Aether!" }],
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### 2. Health Check

**GET** `/health`

Returns the operational status of the server and the hardware device being used.

**Response:**

```json
{
  "status": "ok",
  "device": "cuda"
}
```

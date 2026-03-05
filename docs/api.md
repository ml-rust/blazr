# API Reference

blazr provides an OpenAI-compatible REST API for text generation and chat completions.

## Base URL

When running locally:
```
http://localhost:8080
```

## Endpoints

### Health Check

**GET** `/health`

Check if the server is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### List Models

**GET** `/v1/models`

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "nano",
      "object": "model",
      "created": 1701388800,
      "owned_by": "oxidizr"
    }
  ]
}
```

---

### Text Completions

**POST** `/v1/completions`

Generate text completion from a prompt.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text prompt |
| `max_tokens` | integer | No | 100 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | No | 40 | Top-k sampling |
| `stream` | boolean | No | false | Stream response tokens (not yet implemented) |
| `model` | string | No | "nano" | Model identifier (currently ignored) |

**Response:**
```json
{
  "id": "cmpl-a1b2c3d4",
  "object": "text_completion",
  "created": 1701388800,
  "model": "nano",
  "choices": [
    {
      "text": " in a land far away, there lived...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 96,
    "total_tokens": 100
  }
}
```

**Finish Reasons:**
- `length` - Maximum token limit reached
- `stop` - Stop sequence encountered (not yet implemented)
- `eos` - End-of-sequence token generated (not yet implemented)

---

### Chat Completions

**POST** `/v1/chat/completions`

Generate chat completion from a conversation.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `messages` | array | Yes | - | Array of message objects |
| `max_tokens` | integer | No | 100 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | No | 40 | Top-k sampling |
| `stream` | boolean | No | false | Stream response tokens (not yet implemented) |
| `model` | string | No | "nano" | Model identifier (currently ignored) |

**Message Object:**
```json
{
  "role": "system" | "user" | "assistant",
  "content": "Message text"
}
```

**Response:**
```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1701388800,
  "model": "nano",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

---

## Chat Format

Messages are formatted using special tokens:

```
<|system|>
You are a helpful assistant.
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.
```

The server automatically appends `<|assistant|>\n` to prompt the model for a response.

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error message description"
}
```

**HTTP Status Codes:**
- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server-side error during inference

---

## Examples

### cURL

```bash
# Text completion
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox",
    "max_tokens": 20,
    "temperature": 0.8
  }'

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a pirate."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

### Python (requests)

```python
import requests

# Text completion
response = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "prompt": "The quick brown fox",
        "max_tokens": 20,
        "temperature": 0.8
    }
)
print(response.json()["choices"][0]["text"])

# Chat completion
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### JavaScript (fetch)

```javascript
// Text completion
const response = await fetch("http://localhost:8080/v1/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "The quick brown fox",
    max_tokens: 20,
    temperature: 0.8
  })
});
const data = await response.json();
console.log(data.choices[0].text);

// Chat completion
const chatResponse = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is 2+2?" }
    ],
    max_tokens: 50
  })
});
const chatData = await chatResponse.json();
console.log(chatData.choices[0].message.content);
```

---

## Rate Limiting

Currently, blazr does not implement rate limiting. For production deployments, consider using a reverse proxy (nginx, Caddy) to add rate limiting.

---

## Streaming (Planned)

Streaming support is planned for future releases. When implemented, set `"stream": true` to receive server-sent events:

```
data: {"id":"cmpl-xxx","object":"text_completion.chunk",...}
data: {"id":"cmpl-xxx","object":"text_completion.chunk",...}
data: [DONE]
```

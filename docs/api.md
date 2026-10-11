# blazr HTTP API

blazr serves text generation, embeddings, reranking, audio, and model management over HTTP.

## Base URL and transport

- Default base URL: `http://localhost:8080`.
- The server binds `0.0.0.0:8080` by default. Change the bind with `--host` and `--port`.
- Pass `--tls-cert` and `--tls-key` together to serve HTTPS.

## Authentication

Authentication is off until you set an API key. Set keys with `--api-key`, the `BLAZR_API_KEY` environment variable, or `--api-key-file`. The key file holds one key per line.

When any key is set, every route except `GET /health` and `GET /metrics` requires this header:

```
Authorization: Bearer <api-key>
```

A missing or unknown key returns `401`. The examples in this document omit the header.

## Errors

The error body depends on the route family.

| Route family                                            | Body shape                                                |
| ------------------------------------------------------- | --------------------------------------------------------- |
| OpenAI-compatible, slots, LoRA, tokenize, rerank, audio | `{"error":{"message":"...","type":"..."}}`                |
| Anthropic (`/v1/messages`, `/v1/messages/count_tokens`) | `{"type":"error","error":{"type":"...","message":"..."}}` |
| Ollama-style (`/api/show`, `/api/delete`, `/api/copy`)  | `{"error":"..."}`                                         |

The `401` body uses the OpenAI shape with `"type":"invalid_api_key"`.

Some `500` responses from `GET /v1/models` and `GET /api/tags` return the error text as a plain body.

## Streaming

Completions, chat, infill, and Anthropic messages stream with `"stream": true`.

- OpenAI-style routes send Server-Sent Events. Each event is one `data:` line holding a JSON chunk.
- The stream ends with `data: [DONE]`. The Anthropic stream has no `[DONE]` line.
- Anthropic events use named `event:` lines, such as `message_start` and `content_block_delta`.
- A generation error mid-stream sends one error event and ends the stream.

The OpenAI-style error event has this body:

```
data: {"error":{"message":"...","type":"server_error"}}
```

Streaming responses do not return `logprobs`. Tool calls are parsed only in non-streaming chat responses.

## Endpoint index

| Method    | Path                         | Purpose                            | Gate          |
| --------- | ---------------------------- | ---------------------------------- | ------------- |
| GET       | `/health`                    | Server status and loaded models    | none, no auth |
| GET       | `/metrics`                   | Prometheus metrics                 | none, no auth |
| GET       | `/v1/models`                 | List models                        | none          |
| GET       | `/v1/models/{model_id}`      | Show one model                     | none          |
| POST      | `/v1/completions`            | Text completion                    | none          |
| POST      | `/v1/chat/completions`       | Chat completion                    | none          |
| POST      | `/v1/responses`              | Responses API                      | none          |
| POST      | `/v1/messages`               | Anthropic Messages API             | none          |
| POST      | `/v1/messages/count_tokens`  | Count Anthropic message tokens     | none          |
| POST      | `/v1/embeddings`             | Embeddings                         | none          |
| POST      | `/rerank`, `POST /v1/rerank` | Rerank documents                   | none          |
| POST      | `/v1/infill`                 | Fill-in-the-middle completion      | none          |
| POST      | `/tokenize`                  | Text to token IDs                  | none          |
| POST      | `/detokenize`                | Token IDs to text                  | none          |
| POST      | `/apply-template`            | Render a chat template             | none          |
| GET, POST | `/api/slots`                 | List or create inference slots     | none          |
| DELETE    | `/api/slots/{slot_id}`       | Free an inference slot             | none          |
| GET, POST | `/v1/lora`                   | List or load LoRA adapters         | none          |
| DELETE    | `/v1/lora/{name}`            | Unload a LoRA adapter              | none          |
| GET       | `/api/tags`                  | List local models                  | none          |
| POST      | `/api/show`                  | Show model details                 | none          |
| GET       | `/api/ps`                    | List loaded models                 | none          |
| DELETE    | `/api/delete`                | Remove a local model               | none          |
| POST      | `/api/copy`                  | Copy a local model                 | none          |
| POST      | `/api/pull`                  | Download a model from Hugging Face | none          |
| POST      | `/v1/audio/speech`           | Text to speech                     | `audio`       |
| POST      | `/v1/audio/transcriptions`   | Speech to text                     | `audio`       |
| POST      | `/v1/audio/translations`     | Speech to English text             | `audio`       |
| GET       | `/v1/audio/voices`           | List voices of a TTS model         | `audio`       |

The `audio` gate means the build must include the `audio` feature. Other builds do not register these routes.

## Shared request fields

### Sampling fields

Completions and chat accept these fields. Infill accepts a subset.

| Field                   | Type             | Default | Meaning                                                                       |
| ----------------------- | ---------------- | ------- | ----------------------------------------------------------------------------- |
| `max_tokens`            | integer          | `2048`  | Maximum tokens to generate. Must be greater than `0`.                         |
| `temperature`           | number           | `1.0`   | Sampling temperature. Range `0` to `2`.                                       |
| `top_p`                 | number           | `1.0`   | Nucleus sampling. Range `0` to `1`.                                           |
| `top_k`                 | integer          | `0`     | Top-k sampling limit.                                                         |
| `min_p`                 | number           | `0.05`  | Minimum probability relative to the top token.                                |
| `repeat_penalty`        | number           | `1.1`   | Repetition penalty.                                                           |
| `frequency_penalty`     | number           | `0.0`   | Frequency penalty.                                                            |
| `presence_penalty`      | number           | `0.0`   | Presence penalty.                                                             |
| `typical_p`             | number           | `0.0`   | Typical sampling.                                                             |
| `seed`                  | integer          | none    | Sampling seed. With `n`, choice `i` uses `seed + i`.                          |
| `stop`                  | array of strings | none    | Stop sequences. Honored in streaming and non-streaming responses.             |
| `logit_bias`            | object           | none    | Map of token ID string to bias. Keys that do not parse as `u32` are dropped.  |
| `logprobs`              | boolean          | `false` | Return token log probabilities.                                               |
| `top_logprobs`          | integer          | `5`     | Alternatives per token. Capped at `20`.                                       |
| `mirostat`              | integer          | `0`     | Mirostat mode.                                                                |
| `mirostat_tau`          | number           | `5.0`   | Mirostat target entropy.                                                      |
| `mirostat_eta`          | number           | `0.1`   | Mirostat learning rate.                                                       |
| `dynatemp_range`        | number           | `0.0`   | Dynamic temperature range.                                                    |
| `dynatemp_exponent`     | number           | `1.0`   | Dynamic temperature exponent.                                                 |
| `dry_multiplier`        | number           | `0.0`   | DRY repetition penalty multiplier.                                            |
| `dry_base`              | integer          | `2`     | DRY base.                                                                     |
| `dry_allowed_length`    | integer          | `0`     | DRY allowed repeat length.                                                    |
| `dry_sequence_breakers` | array of strings | none    | DRY sequence breakers.                                                        |
| `grammar`               | string           | none    | GBNF grammar that constrains output.                                          |
| `response_format`       | object           | none    | Structured output. See [Structured output](#structured-output).               |
| `keep_alive`            | string           | none    | Time the model stays loaded after the request. See [keep_alive](#keep_alive). |
| `lora_adapter`          | string           | none    | Name of a loaded LoRA adapter to apply. See [LoRA adapters](#lora-adapters).  |
| `user`                  | string           | none    | Caller identifier, passed to generation metrics.                              |

Each field the request omits takes its value from the `generation` section of `~/.blazr/config.yaml`. The table shows the built-in values. Changes to `generation` apply to the next request without a restart.

### Structured output

`response_format` takes one of these types.

| `type`        | Effect                                                                |
| ------------- | --------------------------------------------------------------------- |
| `text`        | Default. No constraint.                                               |
| `json_object` | Enables JSON mode.                                                    |
| `json_schema` | Converts `json_schema` to a GBNF grammar and constrains output to it. |

- An explicit `grammar` field takes priority over `response_format`.
- A `json_schema` that fails conversion applies no grammar. The request does not fail.

### keep_alive

`keep_alive` sets how long the model stays loaded after the request. It accepts these forms.

| Value                   | Meaning                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `"0"`                   | Unload immediately.                                        |
| `"-1"`                  | Keep loaded indefinitely.                                  |
| `"30s"`, `"5m"`, `"1h"` | Keep loaded for the duration.                              |
| `"300"`                 | Keep loaded for 300 seconds. A bare number counts seconds. |

## Completions

`POST /v1/completions`

Generates text from a prompt.

| Field             | Type              | Default  | Meaning                                                                 |
| ----------------- | ----------------- | -------- | ----------------------------------------------------------------------- |
| `model`           | string            | required | Model name.                                                             |
| `prompt`          | string            | required | Prompt text.                                                            |
| `stream`          | boolean           | `false`  | Stream the response as SSE.                                             |
| `n`               | integer           | `1`      | Number of choices. Non-streaming only.                                  |
| `echo`            | boolean           | `false`  | Prepend the prompt to each choice. Non-streaming only.                  |
| `suffix`          | string            | none     | Append text to each choice. Non-streaming only.                         |
| `context`         | array of integers | none     | Token IDs decoded and prepended to the prompt.                          |
| `allowed_special` | array of strings  | none     | Control tokens that `prompt` encodes as real tokens. Absent means none. |
| Sampling fields   |                   |          | See [Sampling fields](#sampling-fields).                                |

Special tokens in `prompt` encode as their literal text by default. Name a token in `allowed_special` to allow it. Any other control token returns `400`.

### Response

| Field                     | Meaning                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `id`                      | `cmpl-` followed by a UUID.                                   |
| `object`                  | `text_completion`.                                            |
| `created`                 | Unix timestamp in seconds.                                    |
| `model`                   | Model name from the request.                                  |
| `choices[].text`          | Generated text.                                               |
| `choices[].index`         | Choice index.                                                 |
| `choices[].finish_reason` | `stop` or `length`.                                           |
| `choices[].logprobs`      | Present when `logprobs` is `true` on the single-request path. |
| `usage`                   | Token counts and timing. See [Usage](#usage).                 |
| `done_reason`             | Finish reason of the last choice.                             |

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

```json
{
  "id": "cmpl-3f2a9c1e-0000-4000-8000-000000000001",
  "object": "text_completion",
  "created": 1767225600,
  "model": "llama-3.2-1b",
  "choices": [{ "text": " Paris.", "index": 0, "finish_reason": "stop" }],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 3,
    "total_tokens": 9,
    "total_duration_ms": 41,
    "prompt_eval_duration_ms": 12,
    "load_duration_ms": 0,
    "tokens_per_second": 73.1
  },
  "done_reason": "stop"
}
```

Example values are illustrative.

### Streaming chunk

With `"stream": true`, each event has this shape. The stream ends with `data: [DONE]`.

```json
{
  "id": "cmpl-3f2a9c1e-0000-4000-8000-000000000001",
  "object": "text_completion.chunk",
  "created": 1767225600,
  "model": "llama-3.2-1b",
  "choices": [
    { "index": 0, "delta": { "text": " Paris" }, "finish_reason": null }
  ]
}
```

The final chunk sets `finish_reason` to `stop` or `length`.

```bash
curl -N http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "prompt": "The capital of France is", "max_tokens": 16, "stream": true}'
```

## Chat completions

`POST /v1/chat/completions`

Generates an assistant reply from a message list.

| Field                                                               | Type              | Default        | Meaning                                                                                      |
| ------------------------------------------------------------------- | ----------------- | -------------- | -------------------------------------------------------------------------------------------- |
| `model`                                                             | string            | required       | Model name.                                                                                  |
| `messages`                                                          | array             | required       | Messages. Must not be empty.                                                                 |
| `system`                                                            | string            | none           | System prompt. Added only when no message has role `system`.                                 |
| `template`                                                          | string            | model template | Built-in template name or `raw`. See [Chat templates](#chat-templates).                      |
| `raw`                                                               | boolean           | `false`        | Join message text with newlines and skip the template.                                       |
| `tools`                                                             | array             | none           | Tool definitions. See [Tools](#tools).                                                       |
| `tool_choice`                                                       | any               | none           | Accepted and not applied.                                                                    |
| `think`                                                             | boolean           | `false`        | Sets the template variable `enable_thinking`. Returns `thinking` on non-streaming responses. |
| `reasoning_effort`                                                  | string            | none           | Sets the template variable `reasoning_effort`.                                               |
| `chat_template_kwargs`                                              | object            | none           | Extra template variables. Passed to the template as given.                                   |
| `reasoning_format`                                                  | string            | server default | `deepseek` or `none`. See [Reasoning output](#reasoning-output).                             |
| `stream`                                                            | boolean           | `false`        | Stream the response as SSE.                                                                  |
| `n`                                                                 | integer           | `1`            | Number of choices. Non-streaming only.                                                       |
| `context`                                                           | array of integers | none           | Token IDs decoded and prepended to the prompt.                                               |
| `lora_adapter`                                                      | string            | none           | Loaded LoRA adapter to apply.                                                                |
| `grammar`, `response_format`, `keep_alive`, `user`, sampling fields |                   |                | See [Shared request fields](#shared-request-fields).                                         |

### Messages

| Field               | Type            | Meaning                                                                          |
| ------------------- | --------------- | -------------------------------------------------------------------------------- |
| `role`              | string          | Message role, such as `system`, `user`, `assistant`, or `tool`.                  |
| `content`           | string or array | Text, or an array of parts. See [Image and audio input](#image-and-audio-input). |
| `tool_calls`        | array           | Tool calls from an earlier assistant turn.                                       |
| `tool_call_id`      | string          | ID of the tool call this `tool` message answers.                                 |
| `name`              | string          | Function name for a `tool` message.                                              |
| `reasoning_content` | string          | Reasoning from an earlier assistant turn. The template decides how to render it. |

### Response

| Field                     | Meaning                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| `id`                      | `chatcmpl-` followed by a UUID.                                                          |
| `object`                  | `chat.completion`.                                                                       |
| `created`                 | Unix timestamp in seconds.                                                               |
| `model`                   | Model name from the request.                                                             |
| `choices[].message`       | Assistant message with `role`, `content`, `reasoning_content`, and `tool_calls`.         |
| `choices[].finish_reason` | `stop`, `length`, or `tool_calls`.                                                       |
| `choices[].logprobs`      | Present when `logprobs` is `true` on the single-request path.                            |
| `thinking`                | First choice reasoning. Present when `think` is `true` and the model produced reasoning. |
| `usage`                   | Token counts and timing. See [Usage](#usage).                                            |
| `done_reason`             | Finish reason of the last choice.                                                        |

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [
      {"role": "system", "content": "Answer in one word."},
      {"role": "user", "content": "What color is the sky on a clear day?"}
    ],
    "max_tokens": 8,
    "temperature": 0
  }'
```

```json
{
  "id": "chatcmpl-6b1d2e3f-0000-4000-8000-000000000002",
  "object": "chat.completion",
  "created": 1767225600,
  "model": "llama-3.2-1b",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Blue." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 31,
    "completion_tokens": 3,
    "total_tokens": 34,
    "total_duration_ms": 58,
    "prompt_eval_duration_ms": 22,
    "load_duration_ms": 0,
    "tokens_per_second": 51.7
  },
  "done_reason": "stop"
}
```

Example values are illustrative.

### Streaming chunk

Each chunk has `object` set to `chat.completion.chunk`. The `delta` holds `role`, `content`, or `reasoning_content`, and only the fields with a value appear.

```json
{
  "id": "chatcmpl-6b1d2e3f-0000-4000-8000-000000000002",
  "object": "chat.completion.chunk",
  "created": 1767225600,
  "model": "llama-3.2-1b",
  "choices": [
    { "index": 0, "delta": { "content": "Blue" }, "finish_reason": null }
  ]
}
```

### Chat templates

The server picks the chat template in this order:

- The request field `template`, when set.
- The template the model ships with.
- A built-in template for the model type.

`template` accepts a built-in name or `raw`. Built-in names:

- `llama3`, `llama`, `llama2`
- `mistral`
- `chatml`, `qwen`, `yi`
- `phi3`, `phi`
- `gemma`, `gemma2`
- `deepseek`
- `generic`
- `raw`

An unknown name returns `400`.

Set `raw` to `true` to skip the template. `raw` joins the message text with `\n`. The server encodes the text with control tokens disabled.

With `raw` set, image parts return `400`.

### Tools

Tool definitions use the OpenAI shape.

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": { "city": { "type": "string" } },
      "required": ["city"]
    }
  }
}
```

- A Jinja chat template receives `tools` and renders them.
- A built-in template receives a system prompt that lists the tools.
- The model calls a tool with `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`. The server also reads an XML form and a bare JSON object with `name` and `arguments`.
- A parsed call returns `finish_reason` `tool_calls`. `message.content` holds the text before the first call. It is `null` when that text is empty.

Each returned call has this shape. The server generates `id`.

```json
{
  "id": "5c0e8f1a2b3d4c5e6f708192a3b4c5d6",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"Kuala Lumpur\"}"
  }
}
```

Send the result back as a `tool` message with `tool_call_id` set to the call `id`.

Tool calls appear only in non-streaming responses.

### Reasoning output

`reasoning_format` sets how the server splits reasoning from the answer.

| Value      | Effect                                                                        |
| ---------- | ----------------------------------------------------------------------------- |
| `deepseek` | Default. Reasoning goes to `reasoning_content`. The answer goes to `content`. |
| `none`     | Raw model output goes to `content`, tags included.                            |

The chat request field overrides the server default. The Responses and Anthropic routes use only the server default. The server default comes from `--reasoning-format`, then `server.reasoning_format`, then `deepseek`.

### Image and audio input

Chat messages accept image and audio parts in `content`.

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Describe this image." },
    {
      "type": "image_url",
      "image_url": { "url": "https://example.com/cat.png" }
    }
  ]
}
```

| Part          | Fields                                   | Source                                                     |
| ------------- | ---------------------------------------- | ---------------------------------------------------------- |
| `image_url`   | `image_url.url`                          | `data:` URI, `http://` or `https://` URL, or plain base64. |
| `input_audio` | `input_audio.data`, `input_audio.format` | Base64 audio.                                              |

- A value that is neither a `data:` URI nor an `http(s)` URL decodes as plain base64. The server does not read local file paths.
- Raw base64 must decode to a PNG, JPEG, WebP, or GIF image. Any other value fails with this message:

  ```
  Invalid image_url: expected a data: URI, an http(s) URL, or base64 of a PNG, JPEG, WebP, or GIF image
  ```

- Base64 that fails to decode fails with `Invalid image_url: expected a data: URI, an http(s) URL, or base64 (<decode error>)`.
- A `http(s)` fetch that returns a non-success status returns `400`.

Image routing depends on the loaded model:

| Model                             | Images                                                      | Audio    |
| --------------------------------- | ----------------------------------------------------------- | -------- |
| Text only                         | `400`. Start the server with `--mmproj` for a vision tower. | `400`    |
| Vision tower (`--mmproj`, qwen35) | Accepted                                                    | `400`    |
| Multimodal model (own encoders)   | Accepted                                                    | Accepted |

Each image marker must appear once per image when a vision tower handles the request.

## Responses API

`POST /v1/responses`

An OpenAI Responses API request. This route does not stream.

| Field               | Type             | Default  | Meaning                                 |
| ------------------- | ---------------- | -------- | --------------------------------------- |
| `model`             | string           | required | Model name.                             |
| `input`             | string or array  | required | User text, or an array of input items.  |
| `instructions`      | string           | none     | System message placed before the input. |
| `max_output_tokens` | integer          | `2048`   | Maximum tokens to generate.             |
| `temperature`       | number           | `1.0`    | Same as the sampling field.             |
| `top_p`             | number           | `1.0`    | Same as the sampling field.             |
| `stop`              | array of strings | none     | Stop sequences.                         |
| `reasoning`         | object           | none     | Parsed and not applied.                 |
| `text`              | object           | none     | Parsed and not applied.                 |
| `metadata`          | object           | none     | Parsed and not applied.                 |

An input item has `role` and `content`. The role `developer` maps to `system`. Content takes a string or parts of these types: `input_text`, `input_image` (with `image_url`), and `input_audio`.

```bash
curl http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "input": "Name one prime number.", "max_output_tokens": 32, "temperature": 0}'
```

```json
{
  "id": "resp_9d8c7b6a-0000-4000-8000-000000000003",
  "object": "response",
  "created_at": 1767225600,
  "model": "llama-3.2-1b",
  "output": [
    {
      "type": "message",
      "id": "msg_1a2b3c4d-0000-4000-8000-000000000004",
      "role": "assistant",
      "content": [{ "type": "output_text", "text": "2" }],
      "status": "completed"
    }
  ],
  "usage": { "input_tokens": 12, "output_tokens": 4, "total_tokens": 16 },
  "status": "completed"
}
```

When the model produces reasoning, `output` holds a `reasoning` item before the `message` item. Its `summary` holds one `summary_text` entry.

## Anthropic Messages API

`POST /v1/messages`

An Anthropic Messages API request.

| Field            | Type                           | Default  | Meaning                                                |
| ---------------- | ------------------------------ | -------- | ------------------------------------------------------ |
| `model`          | string                         | required | Model name.                                            |
| `max_tokens`     | integer                        | required | Maximum tokens to generate. Must be greater than `0`.  |
| `messages`       | array                          | required | Messages with `role` and `content`. Must not be empty. |
| `system`         | string or array of text blocks | none     | System prompt.                                         |
| `stop_sequences` | array of strings               | none     | Stop sequences.                                        |
| `stream`         | boolean                        | `false`  | Stream the response as named SSE events.               |
| `temperature`    | number                         | `1.0`    | Range `0` to `2`.                                      |
| `top_p`          | number                         | `1.0`    | Range `0` to `1`.                                      |
| `top_k`          | integer                        | `0`      | Top-k sampling.                                        |

`content` takes a string or an array of blocks. The server reads two block types.

- `{"type": "text", "text": "..."}`
- `{"type": "image", "source": {...}}` with a source of `{"type": "base64", "media_type": "...", "data": "..."}` or `{"type": "url", "url": "..."}`.

A source of any other type returns `400`.

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-1b",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Name one prime number."}],
    "temperature": 0
  }'
```

```json
{
  "id": "msg_5e6f7a8b-0000-4000-8000-000000000005",
  "type": "message",
  "role": "assistant",
  "content": [{ "type": "text", "text": "2" }],
  "model": "llama-3.2-1b",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": { "input_tokens": 14, "output_tokens": 3 }
}
```

- `stop_reason` is `end_turn` or `max_tokens`.
- `stop_sequence` is always `null`.
- A reasoning split adds a `thinking` block before the `text` block.

### Streaming events

With `"stream": true`, the server sends these events in order.

| Event                 | Purpose                                     |
| --------------------- | ------------------------------------------- |
| `message_start`       | Starts the message.                         |
| `content_block_start` | Starts a `text` or `thinking` block.        |
| `content_block_delta` | Carries a `text_delta` or `thinking_delta`. |
| `content_block_stop`  | Ends a block.                               |
| `message_delta`       | Carries `stop_reason`.                      |
| `message_stop`        | Ends the message.                           |
| `error`               | Sent on a generation error.                 |

### Count tokens

`POST /v1/messages/count_tokens` takes the same body as `POST /v1/messages`. It returns the input token count only.

```bash
curl http://localhost:8080/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "max_tokens": 1, "messages": [{"role": "user", "content": "Hello"}]}'
```

```json
{ "input_tokens": 9 }
```

## Embeddings

`POST /v1/embeddings`

| Field             | Type                               | Default  | Meaning                                                                  |
| ----------------- | ---------------------------------- | -------- | ------------------------------------------------------------------------ |
| `model`           | string                             | required | Model name.                                                              |
| `input`           | string, array of strings, or parts | required | Text to embed. Images are accepted only by a registered vision embedder. |
| `encoding_format` | string                             | `float`  | `float` or `base64`.                                                     |
| `pooling`         | string                             | `mean`   | `mean`, `cls`, `last`, or `none`.                                        |
| `normalize`       | boolean                            | `false`  | L2-normalize each vector.                                                |
| `user`            | string                             | none     | Caller identifier.                                                       |

Image input needs a vision embedder registered for the model with `--vision-model NAME=PATH`. Without one, the server returns `400`.

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-small", "input": ["first text", "second text"], "normalize": true}'
```

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.012, -0.034, 0.177] },
    { "object": "embedding", "index": 1, "embedding": [0.008, -0.021, 0.163] }
  ],
  "model": "bge-small",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

The example vectors are shortened. Real vectors carry the full model dimension.

## Rerank

`POST /rerank` and `POST /v1/rerank`

Scores documents against a query.

| Field              | Type    | Default  | Meaning                                                     |
| ------------------ | ------- | -------- | ----------------------------------------------------------- |
| `model`            | string  | required | Model name.                                                 |
| `query`            | string  | required | Search query.                                               |
| `documents`        | array   | required | Strings, or objects with a `text` field. Must not be empty. |
| `top_n`            | integer | all      | Number of results to return.                                |
| `return_documents` | boolean | `true`   | Include the document text in each result.                   |

```bash
curl http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-reranker", "query": "capital of France", "documents": ["Paris is in France.", "Bananas are yellow."], "top_n": 1}'
```

```json
{
  "object": "list",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.94,
      "document": { "text": "Paris is in France." }
    }
  ],
  "model": "bge-reranker",
  "usage": { "total_tokens": 14 }
}
```

`document` is absent when `return_documents` is `false`.

## Infill

`POST /v1/infill`

Fills code between a prefix and a suffix. The model tokenizer must define the fill-in-the-middle tokens. Otherwise the server returns `400`.

| Field                                    | Type              | Default           | Meaning                                                         |
| ---------------------------------------- | ----------------- | ----------------- | --------------------------------------------------------------- |
| `model`                                  | string            | required          | Model name.                                                     |
| `prefix`                                 | string            | required          | Text before the cursor.                                         |
| `suffix`                                 | string            | `""`              | Text after the cursor.                                          |
| `max_tokens`                             | integer           | `2048`            | Maximum tokens to generate.                                     |
| `temperature`, `top_p`, `top_k`, `min_p` | number or integer | sampling defaults | See [Sampling fields](#sampling-fields).                        |
| `stream`                                 | boolean           | `false`           | Stream the response as SSE with `text_completion.chunk` events. |
| `stop`                                   | array of strings  | none              | Stop sequences.                                                 |
| `seed`                                   | integer           | none              | Sampling seed.                                                  |
| `keep_alive`                             | string            | none              | See [keep_alive](#keep_alive).                                  |
| `user`                                   | string            | none              | Caller identifier.                                              |

```bash
curl http://localhost:8080/v1/infill \
  -H "Content-Type: application/json" \
  -d '{"model": "codellama-7b", "prefix": "def add(a, b):\n    return ", "suffix": "\n", "max_tokens": 16, "temperature": 0}'
```

```json
{
  "id": "fim-7d8e9f0a-0000-4000-8000-000000000006",
  "object": "text_completion",
  "created": 1767225600,
  "model": "codellama-7b",
  "text": "a + b",
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 4,
    "total_tokens": 18,
    "total_duration_ms": 36,
    "prompt_eval_duration_ms": 10,
    "load_duration_ms": 0,
    "tokens_per_second": 111.1
  },
  "finish_reason": "stop"
}
```

The server builds the prompt in prefix, suffix, middle order. The response `text` holds the generated middle.

## Tokenize, detokenize, and apply-template

### POST /tokenize

| Field             | Type             | Default  | Meaning                                                                  |
| ----------------- | ---------------- | -------- | ------------------------------------------------------------------------ |
| `model`           | string           | required | Model name.                                                              |
| `content`         | string           | required | Text to encode.                                                          |
| `allowed_special` | array of strings | none     | Control tokens that `content` encodes as real tokens. Absent means none. |

```bash
curl http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "content": "Hello world"}'
```

```json
{ "tokens": [9906, 1917] }
```

### POST /detokenize

| Field    | Type              | Meaning              |
| -------- | ----------------- | -------------------- |
| `model`  | string            | Model name.          |
| `tokens` | array of integers | Token IDs to decode. |

```bash
curl http://localhost:8080/detokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "tokens": [9906, 1917]}'
```

```json
{ "content": "Hello world" }
```

### POST /apply-template

Renders a chat template without running inference.

| Field                   | Type    | Default        | Meaning                             |
| ----------------------- | ------- | -------------- | ----------------------------------- |
| `model`                 | string  | required       | Model name.                         |
| `messages`              | array   | required       | Messages, as in chat completions.   |
| `template`              | string  | model template | Built-in template name or `raw`.    |
| `tools`                 | array   | none           | Tool definitions.                   |
| `add_generation_prompt` | boolean | `true`         | Open the assistant turn at the end. |
| `chat_template_kwargs`  | object  | none           | Extra template variables.           |

```bash
curl http://localhost:8080/apply-template \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "Hi"}]}'
```

```json
{
  "prompt": "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}
```

## Models

### GET /v1/models

Lists text models and resident vision, ASR, and TTS models. A text model wins a name collision.

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.2-1b",
      "object": "model",
      "created": 0,
      "owned_by": "local"
    }
  ]
}
```

### GET /v1/models/{model_id}

Returns one model in the same entry shape. Returns `404` when the model is not listed.

## Inference slots

Slots reserve an inference context for a model.

### POST /api/slots

| Field   | Type   | Meaning     |
| ------- | ------ | ----------- |
| `model` | string | Model name. |

Returns `201` with the slot. Returns `503` with type `slot_allocation_failed` when allocation fails.

```bash
curl http://localhost:8080/api/slots \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b"}'
```

```json
{ "id": "slot-1", "model": "llama-3.2-1b", "status": "active" }
```

### GET /api/slots

Returns an array of slots. Each entry has `id`, `model`, `total_tokens`, and `idle_seconds`.

### DELETE /api/slots/{slot_id}

Frees the slot. Returns `204` with no body. Returns `404` with type `slot_not_found` when the slot does not exist.

## LoRA adapters

Adapters use the Hugging Face PEFT format. The model loads on first use.

### POST /v1/lora

| Field   | Type   | Meaning                                                      |
| ------- | ------ | ------------------------------------------------------------ |
| `model` | string | Base model name.                                             |
| `name`  | string | Name for the adapter. Use it as `lora_adapter` in a request. |
| `path`  | string | Adapter directory, or an `adapter_model.safetensors` file.   |

```bash
curl http://localhost:8080/v1/lora \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b", "name": "sql", "path": "/adapters/sql"}'
```

```json
{ "status": "ok", "model": "llama-3.2-1b", "adapter": "sql" }
```

Returns `422` when the adapter fails to load. Returns `404` when the model is not found.

### GET /v1/lora

Requires the query parameter `model`.

```bash
curl "http://localhost:8080/v1/lora?model=llama-3.2-1b"
```

```json
{ "model": "llama-3.2-1b", "adapters": [{ "name": "sql" }] }
```

### DELETE /v1/lora/{name}

Requires the query parameter `model`. Returns `204` on success. Returns `404` with type `lora_not_found` when the adapter is not loaded for the model.

```bash
curl -X DELETE "http://localhost:8080/v1/lora/sql?model=llama-3.2-1b"
```

## Model management

These routes follow the Ollama API shape. They act on local model directories. Example values are illustrative.

### GET /api/tags

Lists local text models with their metadata.

```json
{
  "models": [
    {
      "name": "llama-3.2-1b",
      "model": "llama-3.2-1b",
      "modified_at": "2026-10-01T12:00:00Z",
      "size": 2471645184,
      "details": {
        "format": "safetensors",
        "family": "llama",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

```bash
curl http://localhost:8080/api/tags
```

### POST /api/show

| Field  | Type   | Meaning     |
| ------ | ------ | ----------- |
| `name` | string | Model name. |

Returns the model template, parameters, and details. `modelfile` is always an empty string. `parameters` lists `num_ctx` from the model config. Returns `404` when the model is not found.

```bash
curl http://localhost:8080/api/show \
  -H "Content-Type: application/json" \
  -d '{"name": "llama-3.2-1b"}'
```

```json
{
  "modelfile": "",
  "parameters": "num_ctx 4096\ntemperature 1.0\ntop_p 1.0",
  "template": "...",
  "details": {
    "format": "",
    "family": "llama",
    "parameter_size": "",
    "quantization_level": ""
  },
  "model_info": {
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 2048,
    "num_layers": 16,
    "max_seq_len": 131072
  }
}
```

The `template` value is a description of the chat template.

### GET /api/ps

Lists models loaded now.

```json
{
  "models": [
    {
      "name": "llama-3.2-1b",
      "model": "llama-3.2-1b",
      "size": 2471645184,
      "expires_at": "...",
      "size_vram": 0,
      "details": {
        "format": "",
        "family": "llama",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

```bash
curl http://localhost:8080/api/ps
```

### DELETE /api/delete

| Field  | Type   | Meaning     |
| ------ | ------ | ----------- |
| `name` | string | Model name. |

Unloads the model, then removes its file or directory. Returns `200` with no body. Returns `404` with `{"error": "model '<name>' not found"}` when the model is not found.

```bash
curl -X DELETE http://localhost:8080/api/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "old-model"}'
```

### POST /api/copy

| Field         | Type   | Meaning                                       |
| ------------- | ------ | --------------------------------------------- |
| `source`      | string | Model name to copy.                           |
| `destination` | string | Name for the copy. Created beside the source. |

Returns `200` with no body. Returns `409` when the destination exists. Returns `404` when the source is not found.

```bash
curl http://localhost:8080/api/copy \
  -H "Content-Type: application/json" \
  -d '{"source": "llama-3.2-1b", "destination": "llama-copy"}'
```

### POST /api/pull

| Field  | Type   | Meaning                                       |
| ------ | ------ | --------------------------------------------- |
| `name` | string | Hugging Face repository, such as `org/model`. |

Downloads the model into `<model_dir>/<model-name>`. The server resolves `model_dir` from `BLAZR_MODEL_DIR`, then the `model_dir` config key, then `./models`. The response is `application/x-ndjson`, one JSON object per line.

```json
{"status":"pulling config.json"}
{"status":"downloaded config.json"}
{"status":"success"}
```

- A file that fails to download emits no `downloaded` line.
- A failed client start emits one `{"error": "..."}` line and ends the stream.
- The download includes `config.json`, `model.safetensors`, the tokenizer files, and each shard in `model.safetensors.index.json`.

```bash
curl http://localhost:8080/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "org/model"}'
```

## Audio

These routes exist only in a build with the `audio` feature.

### POST /v1/audio/speech

Synthesizes speech from text. Returns audio bytes.

| Field             | Type    | Default       | Meaning                                               |
| ----------------- | ------- | ------------- | ----------------------------------------------------- |
| `model`           | string  | required      | Loaded TTS model name.                                |
| `input`           | string  | required      | Text to speak. Must not be empty.                     |
| `voice`           | string  | model default | Voice ID. Unknown IDs return `400`.                   |
| `response_format` | string  | `wav`         | `wav`, `pcm`, `f32`, `mp3`, `opus`, `aac`, or `flac`. |
| `speed`           | number  | `1.0`         | Range `0.25` to `4.0`.                                |
| `stream`          | boolean | `false`       | Stream the audio as it renders.                       |

Only `wav`, `pcm`, and `f32` encode natively. Other formats return `501`.

The model default voice is `zero-shot` on VoxCPM2 and `alloy` on Kokoro.

With `stream` set, the body begins with a WAV header that has unknown-length size fields. For `pcm` it has no header. Sample chunks follow. Errors before the first chunk return the same status as the buffered path. An engine error after the first chunk ends the connection early.

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello from blazr.", "voice": "alloy", "response_format": "wav"}' \
  --output speech.wav
```

### POST /v1/audio/transcriptions

Transcribes speech. The request is `multipart/form-data`.

| Field                       | Type   | Default   | Meaning                                          |
| --------------------------- | ------ | --------- | ------------------------------------------------ |
| `file`                      | file   | required  | Audio file. Must not be empty. Limit 25 MB.      |
| `model`                     | string | required  | Loaded ASR model name.                           |
| `language`                  | string | detected  | Language code. Detected when omitted.            |
| `response_format`           | string | `json`    | `json`, `text`, `verbose_json`, `vtt`, or `srt`. |
| `temperature`               | number | none      | Range `0.0` to `1.0`.                            |
| `timestamp_granularities[]` | string | `segment` | `segment` or `word`. One field per value.        |
| `task`                      | string | none      | `translate` switches to translation.             |

The server-wide `max_body_size` applies to this request too. The default is 10 MB. Raise it for uploads above that size.

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=whisper-small" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment"
```

```json
{
  "text": "Welcome to the meeting.",
  "language": "en",
  "duration": 2.4,
  "segments": [
    { "id": 0, "start": 0.0, "end": 2.4, "text": "Welcome to the meeting." }
  ]
}
```

`json` returns only `{"text": "..."}`. `text`, `vtt`, and `srt` return plain text bodies. `words` appears in `verbose_json` when `timestamp_granularities[]` includes `word`.

### POST /v1/audio/translations

Same fields as transcriptions. This route translates the speech to English. The fields `task` and `timestamp_granularities[]` do not apply.

### GET /v1/audio/voices

| Query parameter | Required | Meaning                |
| --------------- | -------- | ---------------------- |
| `model`         | yes      | Loaded TTS model name. |

```bash
curl "http://localhost:8080/v1/audio/voices?model=kokoro"
```

```json
{
  "object": "list",
  "model": "kokoro",
  "data": [{ "id": "af_alloy", "name": "alloy", "language": "en-us" }]
}
```

## Health

`GET /health` returns server status. It needs no auth. The example values are illustrative.

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "loaded_models": ["llama-3.2-1b"],
  "data_parallel_replicas": 1
}
```

`data_parallel_replicas` is the number of copies of the start model that serve requests. It is `1` without `inference.data_parallel` or `--data-parallel`.

`gpu_memory` appears only in a CUDA build when the device reports memory. Its fields are `used_bytes`, `free_bytes`, `total_bytes`, `used_gb`, and `total_gb`.

## Metrics

`GET /metrics` returns Prometheus text format with `Content-Type: text/plain; version=0.0.4; charset=utf-8`. It needs no auth.

```bash
curl http://localhost:8080/metrics
```

## Usage

Completions, chat, and infill return this `usage` object. Responses and Anthropic return their own token fields. The `usage` object has these fields.

| Field                     | Meaning                                                    |
| ------------------------- | ---------------------------------------------------------- |
| `prompt_tokens`           | Prompt tokens.                                             |
| `completion_tokens`       | Generated tokens.                                          |
| `total_tokens`            | Sum of prompt and completion tokens.                       |
| `total_duration_ms`       | Request time in milliseconds. Omitted when zero.           |
| `prompt_eval_duration_ms` | Prompt processing time in milliseconds. Omitted when zero. |
| `load_duration_ms`        | Model lookup time in milliseconds. Omitted when zero.      |
| `tokens_per_second`       | Completion tokens per second. Omitted when zero.           |

## Concurrency and limits

| Config key                | Default | Behavior                                                               |
| ------------------------- | ------- | ---------------------------------------------------------------------- |
| `max_concurrent_requests` | `16`    | Extra requests wait for a free slot. The server does not return `429`. |
| `request_timeout_secs`    | `300`   | Requests running longer return `408`.                                  |
| `max_body_size`           | 10 MB   | Larger request bodies return `413`.                                    |
| `max_inflight_tokens`     | `0`     | In-flight token budget. `0` disables the budget.                       |

`max_inflight_tokens` counts each request as its prompt tokens plus `max_tokens`. A request that exceeds the budget returns `503` with a `Retry-After: 5` header.

`GET /health` and `GET /metrics` skip the `max_concurrent_requests` limit. Probes and scrapes answer while the server is saturated.

## Status codes

| Status | Error type                                                                     | Returned when                                                                                                                  |
| ------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `200`  |                                                                                | Success.                                                                                                                       |
| `201`  |                                                                                | Slot created.                                                                                                                  |
| `204`  |                                                                                | Slot freed, LoRA adapter unloaded.                                                                                             |
| `400`  | `invalid_request_error`                                                        | Invalid field value, empty input, unknown template, unknown voice, control token in text, image input the model cannot handle. |
| `401`  | `invalid_api_key`                                                              | Missing or unknown bearer key.                                                                                                 |
| `404`  | `invalid_request_error`, `not_found_error`, `slot_not_found`, `lora_not_found` | Model, slot, adapter, or source not found.                                                                                     |
| `408`  |                                                                                | Request exceeded `request_timeout_secs`.                                                                                       |
| `409`  |                                                                                | `POST /api/copy` destination exists.                                                                                           |
| `413`  |                                                                                | Body exceeds `max_body_size`, or an audio file exceeds 25 MB.                                                                  |
| `422`  |                                                                                | LoRA adapter failed to load.                                                                                                   |
| `500`  | `server_error`                                                                 | Generation, embedding, detokenization, or synthesis failed.                                                                    |
| `501`  | `not_implemented`                                                              | Speech `response_format` is not native.                                                                                        |
| `503`  | `server_overloaded`                                                            | In-flight token budget exceeded. Includes `Retry-After: 5`.                                                                    |
| `503`  | `slot_allocation_failed`                                                       | Slot allocation failed.                                                                                                        |
| `503`  | `not_implemented`                                                              | TTS model has no synthesis engine.                                                                                             |

## Source of truth

Routes come from `src/server/routes.rs`. Field names, defaults, and status codes come from the handler and type files under `src/server/`. Example values are illustrative.

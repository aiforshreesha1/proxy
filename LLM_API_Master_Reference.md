# 🧠 LLM API Master Reference
## Anthropic Claude · OpenAI GPT · Google Gemini
### Complete JSON Formats · System Injection · Skills · Agents · Reasoning · Multimodal

> **Purpose:** Any developer (or LLM) can use this document to understand exactly how each provider's API works at the raw HTTP/JSON level, how system prompts / skills / CLAUDE.md are injected, and how to convert any feature from one format to another.

---

## TABLE OF CONTENTS

1. [API Anatomy Overview — All 3 Providers](#1-api-anatomy-overview)
2. [Basic Text Generation](#2-basic-text-generation)
3. [System Prompt / Instructions Injection](#3-system-prompt--instructions-injection)
4. [CLAUDE.md / SKILL.md / agent.md — How They're Injected](#4-claudemd--skillmd--agentmd-injection)
5. [Multi-turn Conversation](#5-multi-turn-conversation)
6. [Function / Tool Calling — Full Loop](#6-function--tool-calling)
7. [Structured / JSON Output](#7-structured--json-output)
8. [Multimodal (Images, Files)](#8-multimodal-images-files)
9. [Extended Thinking / Hybrid Reasoning](#9-extended-thinking--hybrid-reasoning)
10. [Streaming Responses](#10-streaming-responses)
11. [Prompt Caching (Anthropic)](#11-prompt-caching-anthropic)
12. [Multi-Agent Orchestration](#12-multi-agent-orchestration)
13. [Format Conversion Cheat Sheet](#13-format-conversion-cheat-sheet)
14. [Response Anatomy — All 3 Providers](#14-response-anatomy)
15. [Error Formats](#15-error-formats)
16. [Headers & Auth Quick Reference](#16-headers--auth-quick-reference)

---

## 1. API ANATOMY OVERVIEW

### Endpoint Map

| Provider | Base URL | Main Endpoint | Auth Header |
|---|---|---|---|
| **Anthropic** | `https://api.anthropic.com` | `POST /v1/messages` | `x-api-key: $KEY` |
| **OpenAI (Chat)** | `https://api.openai.com` | `POST /v1/chat/completions` | `Authorization: Bearer $KEY` |
| **OpenAI (Responses)** | `https://api.openai.com` | `POST /v1/responses` | `Authorization: Bearer $KEY` |
| **Google Gemini** | `https://generativelanguage.googleapis.com` | `POST /v1beta/models/{model}:generateContent` | `x-goog-api-key: $KEY` |
| **Google Vertex** | `https://{region}-aiplatform.googleapis.com` | `POST /v1/projects/{proj}/locations/{loc}/publishers/google/models/{model}:generateContent` | `Authorization: Bearer $(gcloud auth)` |

### Fundamental Structural Difference

```
ANTHROPIC:
  system  → top-level string (separate from messages)
  messages → [{role: "user"/"assistant", content: ...}]

OPENAI (Chat Completions):
  messages → [{role: "system"/"developer"/"user"/"assistant"/"tool", content: ...}]
  (system is INSIDE messages array)

OPENAI (Responses API - newer):
  instructions → top-level string (like system)
  input → [{role: "developer"/"user"/"assistant", content: ...}] OR plain string

GOOGLE GEMINI:
  system_instruction → top-level {parts: [{text: "..."}]}
  contents → [{role: "user"/"model", parts: [{text: "..."}]}]
  (roles are "user"/"model" — NOT "assistant")
```

---

## 2. BASIC TEXT GENERATION

### Anthropic — `POST /v1/messages`

**Request:**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum entanglement in simple terms."
    }
  ]
}
```

**Headers:**
```
x-api-key: $ANTHROPIC_API_KEY
anthropic-version: 2023-06-01
content-type: application/json
```

**Response:**
```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Quantum entanglement is a phenomenon where...",
      "citations": null
    }
  ],
  "model": "claude-opus-4-6",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 120,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**Accessing text:** `response["content"][0]["text"]`

---

### OpenAI Chat Completions — `POST /v1/chat/completions`

**Request:**
```json
{
  "model": "gpt-4o",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum entanglement in simple terms."
    }
  ]
}
```

**Headers:**
```
Authorization: Bearer $OPENAI_API_KEY
content-type: application/json
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "gpt-4o-2024-11-20",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum entanglement is a phenomenon where...",
        "refusal": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135,
    "prompt_tokens_details": { "cached_tokens": 0 },
    "completion_tokens_details": { "reasoning_tokens": 0 }
  }
}
```

**Accessing text:** `response["choices"][0]["message"]["content"]`

---

### OpenAI Responses API — `POST /v1/responses` (Newer, Recommended)

**Request:**
```json
{
  "model": "gpt-5",
  "max_output_tokens": 1024,
  "input": "Explain quantum entanglement in simple terms."
}
```

**Response:**
```json
{
  "id": "resp_689a0cf9...",
  "object": "response",
  "created_at": 1754926332.0,
  "model": "gpt-5",
  "output": [
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Quantum entanglement is a phenomenon where..."
        }
      ],
      "status": "completed"
    }
  ],
  "usage": {
    "input_tokens": 15,
    "output_tokens": 120,
    "total_tokens": 135
  }
}
```

**Accessing text:** `response["output"][0]["content"][0]["text"]`  
Or with SDK shortcut: `response.output_text`

---

### Google Gemini — `POST /v1beta/models/{model}:generateContent`

**Request:**
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Explain quantum entanglement in simple terms."
        }
      ]
    }
  ]
}
```

**URL:** `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`

**Headers:**
```
x-goog-api-key: $GEMINI_API_KEY
content-type: application/json
```

**Response:**
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Quantum entanglement is a phenomenon where..."
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0,
      "safetyRatings": [...]
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 120,
    "totalTokenCount": 130
  }
}
```

**Accessing text:** `response["candidates"][0]["content"]["parts"][0]["text"]`

---

## 3. SYSTEM PROMPT / INSTRUCTIONS INJECTION

> **This is the key mechanism:** system prompts, CLAUDE.md content, SKILL.md content — they ALL get injected here.

### Anthropic — Top-Level `system` Field

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "system": "You are a senior Python engineer. Always write type hints. Use descriptive variable names. When reviewing code, check for security vulnerabilities first.",
  "messages": [
    {
      "role": "user",
      "content": "Review this function: def add(a,b): return a+b"
    }
  ]
}
```

**System can also be an ARRAY of content blocks (for caching):**
```json
{
  "system": [
    {
      "type": "text",
      "text": "You are a senior Python engineer...",
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "messages": [...]
}
```

---

### OpenAI Chat Completions — `system` Role Inside Messages

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a senior Python engineer. Always write type hints."
    },
    {
      "role": "user",
      "content": "Review this function: def add(a,b): return a+b"
    }
  ]
}
```

> **Note for reasoning models (o3, gpt-5):** Use `"role": "developer"` instead of `"system"`. They behave the same but `developer` is the preferred role for o-series models.

```json
{
  "model": "o3",
  "messages": [
    {
      "role": "developer",
      "content": "You are a senior Python engineer. Always write type hints."
    },
    {
      "role": "user",
      "content": "Review this function."
    }
  ]
}
```

---

### OpenAI Responses API — Top-Level `instructions`

```json
{
  "model": "gpt-5",
  "instructions": "You are a senior Python engineer. Always write type hints.",
  "input": [
    {
      "role": "user",
      "content": "Review this function: def add(a,b): return a+b"
    }
  ]
}
```

---

### Google Gemini — Top-Level `system_instruction`

```json
{
  "system_instruction": {
    "parts": [
      {
        "text": "You are a senior Python engineer. Always write type hints. Use descriptive variable names."
      }
    ]
  },
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Review this function: def add(a,b): return a+b"
        }
      ]
    }
  ]
}
```

---

## 4. CLAUDE.md / SKILL.md / agent.md INJECTION

### What Are These Files?

| File | Scope | Purpose | Where Loaded From |
|---|---|---|---|
| `CLAUDE.md` | Project-level | Project context, conventions, architecture notes, persistent memory | `.claude/CLAUDE.md`, `~/.claude/CLAUDE.md` |
| `SKILL.md` | Tool-level | Step-by-step instructions for a specific capability (e.g., how to make PPTX files) | `.claude/skills/*/SKILL.md`, `/mnt/skills/*/SKILL.md` |
| `agent.md` / subagent `.md` | Agent-level | Defines a specialized subagent's system prompt, tools, model | `.claude/agents/*.md` |

---

### How CLAUDE.md Gets Injected Into the API

**CLAUDE.md is NOT a special API parameter.** At the raw API level, its content is read from disk and appended/prepended into the `system` field. Here is the exact flow:

```
1. Claude Code / Agent SDK reads CLAUDE.md from disk
2. Concatenates with base system prompt
3. Final combined text → sent as "system" field in /v1/messages
```

**At the raw JSON level, a CLAUDE.md-loaded request looks like:**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 8096,
  "system": "You are Claude Code, an AI coding assistant...\n\n[CLAUDE.md CONTENT BELOW]\n# My Project\n## Tech Stack\n- Python 3.11, FastAPI, PostgreSQL\n## Code Conventions\n- Always use async/await\n- Type hints required\n## Architecture\n- Services in /app/services/\n- Models in /app/models/\n[END CLAUDE.md]\n",
  "messages": [
    {
      "role": "user",
      "content": "Add a new endpoint for user registration."
    }
  ]
}
```

**Agent SDK code that controls this behavior:**
```python
# Python Agent SDK
from anthropic import Anthropic

# CLAUDE.md is loaded when you specify setting_sources
result = query(
    prompt="Add user registration endpoint",
    options={
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code"  # Claude Code's full system prompt
        },
        "setting_sources": ["project"]  # ← This triggers CLAUDE.md loading
    }
)
```

> **Key insight:** Without `setting_sources: ["project"]`, CLAUDE.md is NOT loaded even with the `claude_code` preset.

---

### How SKILL.md Gets Injected

**SKILL.md is injected as a user message** (not system prompt) into the conversation when the Skill tool is invoked. This is a "two-message pattern":

```
Turn 1 (before skill):
  user: "Create a PowerPoint presentation about AI trends"

Skill invocation happens:
  → System reads /mnt/skills/pptx/SKILL.md
  → Injects SKILL.md content as a NEW user message

Turn 2 (after skill injection — what the model actually sees):
  user: "Create a PowerPoint presentation about AI trends"
  assistant: [skill loaded message acknowledging]
  user: "[SKILL.md CONTENT]\n# PPTX Skill\n## How to create PowerPoint files\n1. Install pptxgenjs...\n\nNow: Create a PowerPoint presentation about AI trends"
```

**At raw API level it looks like:**
```json
{
  "model": "claude-sonnet-4-6",
  "system": "[base system prompt]",
  "messages": [
    {
      "role": "user",
      "content": "Create a PowerPoint about AI trends"
    },
    {
      "role": "assistant",
      "content": "I'll load the PPTX skill first to ensure high-quality output."
    },
    {
      "role": "user",
      "content": "# PPTX SKILL INSTRUCTIONS\n\n## Overview\nUse pptxgenjs to create slides...\n\n## Steps\n1. Install: npm install pptxgenjs\n2. Create slide deck...\n\n[full SKILL.md content here]\n\nNow please create the PowerPoint about AI trends."
    }
  ]
}
```

---

### How agent.md (Subagent Definition) Works

Subagent `.md` files are stored in `.claude/agents/` and follow this format:

```markdown
---
name: code-reviewer
description: Reviews code for quality, security, and best practices. Use when code review is needed.
tools: Read, Grep, Glob
model: claude-sonnet-4-6
---

You are an expert code reviewer with 15 years of experience.

## Review Checklist
1. Security vulnerabilities (SQL injection, XSS, etc.)
2. Performance issues
3. Code style and maintainability
4. Error handling

Always provide specific line numbers and actionable suggestions.
```

**When the parent agent spawns this subagent, at the API level it creates a BRAND NEW /v1/messages call:**

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 8096,
  "system": "You are an expert code reviewer with 15 years of experience.\n\n## Review Checklist\n1. Security vulnerabilities...\n[rest of agent.md content]",
  "tools": [
    { "name": "Read", ... },
    { "name": "Grep", ... },
    { "name": "Glob", ... }
  ],
  "messages": [
    {
      "role": "user",
      "content": "[Task description passed from parent agent]"
    }
  ]
}
```

The subagent runs independently in its own context window and returns results to the parent.

---

### Dynamic Modification — When CLAUDE.md Changes

When you modify CLAUDE.md:
- The file on disk changes
- The NEXT API call from the Agent SDK re-reads the file
- The new content replaces the old content in the `system` field
- Previous conversation history is NOT affected (already sent)
- Only NEW requests pick up the change

This is why CLAUDE.md is called "persistent memory" — it persists across sessions by living on disk, and the SDK reads it fresh each time.

---

## 5. MULTI-TURN CONVERSATION

### Anthropic

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    { "role": "user", "content": "What is 2+2?" },
    { "role": "assistant", "content": "2+2 equals 4." },
    { "role": "user", "content": "What about 3+3?" }
  ]
}
```

> ⚠️ Anthropic is stateless — you MUST send the entire history each turn.

---

### OpenAI Chat Completions

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is 2+2?" },
    { "role": "assistant", "content": "2+2 equals 4." },
    { "role": "user", "content": "What about 3+3?" }
  ]
}
```

---

### OpenAI Responses API (Stateful Option)

```json
{
  "model": "gpt-5",
  "store": true,
  "previous_response_id": "resp_abc123",
  "input": "What about 3+3?"
}
```

> The Responses API can manage state server-side via `store: true` and `previous_response_id`. This avoids sending full history each turn.

---

### Google Gemini

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "What is 2+2?" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "2+2 equals 4." }]
    },
    {
      "role": "user",
      "parts": [{ "text": "What about 3+3?" }]
    }
  ]
}
```

> ⚠️ Gemini uses `"model"` not `"assistant"` for AI turns.

---

## 6. FUNCTION / TOOL CALLING

### The Tool Call Loop (All Providers)

```
1. Send request with tool definitions
2. Model responds with a tool_call (not text)
3. YOU execute the function locally
4. Send tool result back to model
5. Model generates final text response
```

---

### Anthropic — Full Tool Call Loop

**Step 1: Define tools & send request**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name, e.g. 'Paris, France'"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    { "role": "user", "content": "What's the weather in Tokyo?" }
  ]
}
```

**Step 2: Model response (tool_use block)**
```json
{
  "id": "msg_01XkAbC",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_01A2B3C4",
      "name": "get_weather",
      "input": {
        "location": "Tokyo, Japan",
        "unit": "celsius"
      }
    }
  ],
  "stop_reason": "tool_use"
}
```

**Step 3: You run the function, send result back**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "tools": [...],
  "messages": [
    { "role": "user", "content": "What's the weather in Tokyo?" },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01A2B3C4",
          "name": "get_weather",
          "input": { "location": "Tokyo, Japan", "unit": "celsius" }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01A2B3C4",
          "content": "{ \"temperature\": 22, \"condition\": \"Partly cloudy\", \"humidity\": 65 }"
        }
      ]
    }
  ]
}
```

**Step 4: Final model response**
```json
{
  "content": [
    {
      "type": "text",
      "text": "The weather in Tokyo is currently 22°C with partly cloudy skies and 65% humidity."
    }
  ],
  "stop_reason": "end_turn"
}
```

---

### OpenAI Chat Completions — Full Tool Call Loop

**Step 1: Define tools**
```json
{
  "model": "gpt-4o",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name, e.g. 'Paris, France'"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location", "unit"],
          "additionalProperties": false
        }
      }
    }
  ],
  "messages": [
    { "role": "user", "content": "What's the weather in Tokyo?" }
  ]
}
```

**Step 2: Model response**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Tokyo, Japan\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

> ⚠️ OpenAI returns `arguments` as a **JSON string**, not a parsed object. You must `json.loads()` it.

**Step 3: Send result back**
```json
{
  "model": "gpt-4o",
  "tools": [...],
  "messages": [
    { "role": "user", "content": "What's the weather in Tokyo?" },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Tokyo, Japan\", \"unit\": \"celsius\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temperature\": 22, \"condition\": \"Partly cloudy\", \"humidity\": 65}"
    }
  ]
}
```

---

### Google Gemini — Full Tool Call Loop

**Step 1: Define tools**
```json
{
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "get_weather",
          "description": "Get current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "City name"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      ]
    }
  ],
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "What's the weather in Tokyo?" }]
    }
  ]
}
```

**Step 2: Model response**
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "functionCall": {
              "name": "get_weather",
              "args": {
                "location": "Tokyo, Japan",
                "unit": "celsius"
              }
            }
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP"
    }
  ]
}
```

> Note: Gemini returns `args` as a **parsed object** (unlike OpenAI which returns a string).

**Step 3: Send result back**
```json
{
  "tools": [...],
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "What's the weather in Tokyo?" }]
    },
    {
      "role": "model",
      "parts": [
        {
          "functionCall": {
            "name": "get_weather",
            "args": { "location": "Tokyo, Japan", "unit": "celsius" }
          }
        }
      ]
    },
    {
      "role": "user",
      "parts": [
        {
          "functionResponse": {
            "name": "get_weather",
            "response": {
              "content": {
                "temperature": 22,
                "condition": "Partly cloudy",
                "humidity": 65
              }
            }
          }
        }
      ]
    }
  ]
}
```

---

### Key Differences in Tool Calling

| Aspect | Anthropic | OpenAI (Chat) | Gemini |
|---|---|---|---|
| Tool definition key | `input_schema` | `function.parameters` | `parameters` |
| Tool wrapper | `tools: [{name, description, input_schema}]` | `tools: [{type: "function", function: {...}}]` | `tools: [{functionDeclarations: [...]}]` |
| Tool call in response | `content[].type == "tool_use"` | `message.tool_calls[].function` | `parts[].functionCall` |
| Tool call ID | `id` on tool_use block | `id` on tool_call | No ID — match by name |
| Arguments format | **Parsed object** | **JSON string** (must parse) | **Parsed object** |
| Tool result role | `"user"` (with tool_result content type) | `"tool"` role | `"user"` (with functionResponse) |
| Forced tool use | `tool_choice: {"type": "tool", "name": "..."}` | `tool_choice: {"type": "function", "function": {"name": "..."}}` | `tool_config: {function_calling_config: {mode: "ANY", allowed_function_names: [...]}}` |

---

## 7. STRUCTURED / JSON OUTPUT

### Anthropic — JSON Schema Output

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 1024,
  "messages": [
    { "role": "user", "content": "Extract person info from: John Smith, 34, engineer at Google" }
  ],
  "output_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "age": { "type": "integer" },
          "job_title": { "type": "string" },
          "company": { "type": "string" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
      }
    }
  }
}
```

> Note: For Claude 4 models, structured outputs are available without beta header. For older models, add header: `anthropic-beta: structured-outputs-2025-11-13`

**Response gives validated JSON in `content[0].text`:**
```json
{
  "content": [{ "type": "text", "text": "{\"name\": \"John Smith\", \"age\": 34, \"job_title\": \"engineer\", \"company\": \"Google\"}" }]
}
```

---

### OpenAI — JSON Schema Output (Chat Completions)

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "user", "content": "Extract person info from: John Smith, 34, engineer at Google" }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "age": { "type": "integer" },
          "job_title": { "type": "string" },
          "company": { "type": "string" }
        },
        "required": ["name", "age", "job_title", "company"],
        "additionalProperties": false
      }
    }
  }
}
```

**Response in `choices[0].message.content` (as JSON string, parse it).**

---

### OpenAI — JSON Schema Output (Responses API)

```json
{
  "model": "gpt-5",
  "input": "Extract person info from: John Smith, 34, engineer at Google",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": { "type": "string" },
          "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
      }
    }
  }
}
```

---

### Google Gemini — JSON Output

```json
{
  "contents": [
    { "role": "user", "parts": [{ "text": "Extract person info from: John Smith, 34, engineer at Google" }] }
  ],
  "generation_config": {
    "response_mime_type": "application/json",
    "response_schema": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" },
        "job_title": { "type": "string" },
        "company": { "type": "string" }
      },
      "required": ["name", "age"]
    }
  }
}
```

**Response text will be a valid JSON string.**

---

## 8. MULTIMODAL (IMAGES, FILES)

### Anthropic — Image in Message

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRgAB..."
          }
        },
        {
          "type": "text",
          "text": "What's in this image?"
        }
      ]
    }
  ]
}
```

**URL-based image (Anthropic also supports URL):**
```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "https://example.com/image.jpg"
  }
}
```

**PDF document:**
```json
{
  "type": "document",
  "source": {
    "type": "base64",
    "media_type": "application/pdf",
    "data": "JVBERi0xLjQK..."
  }
}
```

---

### OpenAI Chat Completions — Image in Message

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg",
            "detail": "high"
          }
        },
        {
          "type": "text",
          "text": "What's in this image?"
        }
      ]
    }
  ]
}
```

**Base64 image:**
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB..."
  }
}
```

---

### OpenAI Responses API — Image Input

```json
{
  "model": "gpt-5",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_image",
          "image_url": "https://example.com/image.jpg",
          "detail": "auto"
        },
        {
          "type": "input_text",
          "text": "What's in this image?"
        }
      ]
    }
  ]
}
```

---

### Google Gemini — Image in Content

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "inline_data": {
            "mime_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRgAB..."
          }
        },
        {
          "text": "What's in this image?"
        }
      ]
    }
  ]
}
```

**URL-based (file URI for GCS):**
```json
{
  "file_data": {
    "mime_type": "image/jpeg",
    "file_uri": "gs://bucket/image.jpg"
  }
}
```

### Multimodal Key Differences

| Aspect | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Image type key | `"type": "image"` | `"type": "image_url"` | `"inline_data"` or `"file_data"` |
| URL source | `source.type: "url", source.url: "..."` | `image_url.url: "https://..."` | `file_data.file_uri: "gs://..."` |
| Base64 source | `source.type: "base64", source.data: "..."` | `image_url.url: "data:image/jpeg;base64,..."` | `inline_data.data: "..."` |
| PDF support | ✅ `type: "document"` | ✅ via Files API | ✅ via Files API |

---

## 9. EXTENDED THINKING / HYBRID REASONING

### Anthropic — Extended Thinking

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [
    {
      "role": "user",
      "content": "Prove that the square root of 2 is irrational."
    }
  ]
}
```

**For Claude Opus 4.6 (newest, use adaptive mode):**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "adaptive"
  },
  "effort": "medium",
  "messages": [...]
}
```

**Response with thinking blocks:**
```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me think through this proof step by step. The proof is by contradiction...",
      "signature": "ErUB...encrypted..."
    },
    {
      "type": "text",
      "text": "To prove √2 is irrational, we use proof by contradiction..."
    }
  ],
  "stop_reason": "end_turn"
}
```

**Rules:**
- `budget_tokens` must be ≥ 1024 and < `max_tokens`
- For Claude 4 models, thinking content is summarized (not full raw thoughts)
- For Opus 4.6/Sonnet 4.6: use `thinking.type: "adaptive"` + `effort` parameter
- Interleaved thinking (thinking between tool calls): add beta header `interleaved-thinking-2025-05-14`

**Interleaved thinking with tools:**
```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 16000,
  "betas": ["interleaved-thinking-2025-05-14"],
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "tools": [...],
  "messages": [...]
}
```

---

### OpenAI — Reasoning Effort (o3, o4-mini, gpt-5)

**Chat Completions API:**
```json
{
  "model": "o3",
  "messages": [
    { "role": "developer", "content": "You are a math expert." },
    { "role": "user", "content": "Prove that the square root of 2 is irrational." }
  ],
  "reasoning_effort": "high",
  "max_completion_tokens": 16000
}
```

> ⚠️ Reasoning models use `max_completion_tokens` NOT `max_tokens`

**Responses API (preferred for reasoning models):**
```json
{
  "model": "gpt-5",
  "input": [
    { "role": "user", "content": "Prove that √2 is irrational." }
  ],
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "max_output_tokens": 16000
}
```

**Response (Responses API with reasoning):**
```json
{
  "output": [
    {
      "type": "reasoning",
      "id": "rs_abc123",
      "summary": [
        {
          "type": "summary_text",
          "text": "I'll prove this by contradiction. Assume √2 = p/q in lowest terms..."
        }
      ]
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "**Proof that √2 is irrational:**\n\nWe use proof by contradiction..."
        }
      ]
    }
  ]
}
```

**Encrypted reasoning (for preserving across tool calls):**
```json
{
  "model": "o3",
  "input": [...],
  "tools": [...],
  "include": ["reasoning.encrypted_content"],
  "store": false
}
```

---

### Google Gemini — Thinking (Gemini 2.5 / 3)

**Gemini 2.5 Flash (thinking_budget):**
```json
{
  "contents": [
    { "role": "user", "parts": [{ "text": "Prove √2 is irrational." }] }
  ],
  "generation_config": {
    "thinking_config": {
      "thinking_budget": 8192
    }
  }
}
```

**Gemini 3 (thinking_level — new):**
```json
{
  "contents": [
    { "role": "user", "parts": [{ "text": "Prove √2 is irrational." }] }
  ],
  "generation_config": {
    "thinking_level": "high"
  }
}
```

**thinking_level values:** `"low"`, `"medium"`, `"high"` (default for Gemini 3)

**Response includes thinking parts:**
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "thought": true,
            "text": "Let me think through this... assuming √2 = p/q..."
          },
          {
            "text": "**Proof:** We proceed by contradiction. Assume √2 is rational..."
          }
        ],
        "role": "model"
      }
    }
  ]
}
```

---

### Reasoning — Cross-Provider Comparison

| Feature | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Reasoning param | `thinking: {type, budget_tokens}` | `reasoning: {effort}` or `reasoning_effort` | `thinking_config: {thinking_budget}` or `thinking_level` |
| Effort levels | `"adaptive"`, `budget_tokens` int | `"low"`, `"medium"`, `"high"` | `0-24576` (budget) or `"low"/"medium"/"high"` |
| Reasoning visible | Thinking blocks in content | Summary only (encrypted full) | Thought parts in response |
| Preserve across calls | Must pass back signature blocks | `include: ["reasoning.encrypted_content"]` | N/A |
| Min budget | 1024 tokens | N/A | 0 |

---

## 10. STREAMING RESPONSES

### Anthropic — Server-Sent Events

**Request:** Add `"stream": true`
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "stream": true,
  "messages": [{ "role": "user", "content": "Tell me a story." }]
}
```

**Event stream format:**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"usage":{"input_tokens":15,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Once"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" upon"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":87}}

event: message_stop
data: {"type":"message_stop"}
```

---

### OpenAI — Server-Sent Events

**Request:** Add `"stream": true`

**Event format:**
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### Google Gemini — Streaming

**URL:** `.../generateContent` → `.../streamGenerateContent`

**Event format:**
```
data: {"candidates":[{"content":{"parts":[{"text":"Once"}],"role":"model"},"finishReason":"","index":0}]}

data: {"candidates":[{"content":{"parts":[{"text":" upon a time"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{...}}
```

---

## 11. PROMPT CACHING (ANTHROPIC)

Cache large, stable content (system prompts, tool definitions, documents) to save tokens.

**Mark content for caching:**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 1024,
  "system": [
    {
      "type": "text",
      "text": "You are a customer service agent with these guidelines:\n[5000 words of guidelines...]\n",
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "messages": [
    { "role": "user", "content": "What is your return policy?" }
  ]
}
```

**Cache tools:**
```json
{
  "tools": [
    {
      "name": "search_database",
      "description": "...",
      "input_schema": { ... },
      "cache_control": { "type": "ephemeral" }
    }
  ]
}
```

**Response shows cache hits:**
```json
{
  "usage": {
    "input_tokens": 15,
    "output_tokens": 45,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 0
  }
}
```

On subsequent calls with same cached content:
```json
{
  "usage": {
    "input_tokens": 15,
    "output_tokens": 45,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 5000
  }
}
```

Cache reads are ~90% cheaper than full tokens. Cache lasts ~5 minutes (ephemeral).

---

## 12. MULTI-AGENT ORCHESTRATION

### Pattern 1: Orchestrator + Subagents (Anthropic API)

The parent agent calls the child agent by making a NEW API call with a specialized system prompt:

**Parent agent code:**
```python
import requests
import json

ANTHROPIC_API_KEY = "sk-ant-..."
HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

def call_subagent(task: str, agent_system_prompt: str, model="claude-sonnet-4-6"):
    """Spawn a subagent with its own system prompt and context."""
    payload = {
        "model": model,
        "max_tokens": 8096,
        "system": agent_system_prompt,
        "messages": [
            {"role": "user", "content": task}
        ]
    }
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=HEADERS,
        json=payload
    )
    return response.json()["content"][0]["text"]

# Code reviewer subagent
code_review_result = call_subagent(
    task="Review this Python function for security issues:\ndef get_user(id):\n    query = f'SELECT * FROM users WHERE id={id}'\n    return db.execute(query)",
    agent_system_prompt="""You are a security-focused code reviewer.
    
## Your Responsibilities
1. Identify SQL injection vulnerabilities
2. Check for proper input validation
3. Look for authentication/authorization issues
4. Rate severity: CRITICAL/HIGH/MEDIUM/LOW

Always respond with a JSON object: {severity, issue, fix, explanation}"""
)
```

---

### Pattern 2: Tool-Based Orchestration (Agent Calls Model as Tool)

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 4096,
  "system": "You are an orchestrator. Use the available tools to delegate work to specialist agents.",
  "tools": [
    {
      "name": "run_code_review",
      "description": "Runs a specialized code review agent on the provided code. Returns review results.",
      "input_schema": {
        "type": "object",
        "properties": {
          "code": { "type": "string", "description": "Code to review" },
          "focus": { "type": "string", "enum": ["security", "performance", "style"] }
        },
        "required": ["code", "focus"]
      }
    },
    {
      "name": "run_test_agent",
      "description": "Runs a testing agent that writes unit tests for the provided code.",
      "input_schema": {
        "type": "object",
        "properties": {
          "code": { "type": "string" },
          "test_framework": { "type": "string", "enum": ["pytest", "unittest", "jest"] }
        },
        "required": ["code"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Review this code and write tests for it:\ndef add(a, b): return a + b"
    }
  ]
}
```

When the orchestrator calls `run_code_review`, your server intercepts it, makes a NEW API call to a specialist agent, and returns the result as a `tool_result`.

---

### Pattern 3: Parallel Subagents

```python
import asyncio
import aiohttp

async def call_agent(session, task, system_prompt):
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": task}]
    }
    async with session.post(
        "https://api.anthropic.com/v1/messages",
        headers=HEADERS,
        json=payload
    ) as resp:
        data = await resp.json()
        return data["content"][0]["text"]

async def run_parallel_agents(code):
    async with aiohttp.ClientSession() as session:
        # Run three agents in parallel
        results = await asyncio.gather(
            call_agent(session, f"Review security:\n{code}", SECURITY_AGENT_PROMPT),
            call_agent(session, f"Review performance:\n{code}", PERF_AGENT_PROMPT),
            call_agent(session, f"Write tests:\n{code}", TEST_AGENT_PROMPT),
        )
    security_review, perf_review, tests = results
    return security_review, perf_review, tests
```

---

### Subagent `.md` File Format (Claude Code)

```markdown
---
name: security-reviewer
description: Reviews code for security vulnerabilities. Use when analyzing code for SQL injection, XSS, authentication issues, or other security concerns.
tools: Read, Grep, Glob
model: claude-opus-4-6
---

You are an expert application security engineer with 20 years of experience.

## Security Checklist
1. **Injection Attacks**: SQL, NoSQL, command, LDAP injection
2. **Authentication/Authorization**: Broken auth, IDOR, privilege escalation
3. **Sensitive Data Exposure**: Unencrypted PII, exposed secrets
4. **XSS**: Reflected, stored, DOM-based cross-site scripting
5. **Dependency Vulnerabilities**: Known CVEs in libraries

## Output Format
Always respond with:
```json
{
  "overall_risk": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
  "vulnerabilities": [
    {
      "type": "...",
      "severity": "...",
      "location": "file.py:line",
      "description": "...",
      "fix": "..."
    }
  ]
}
```
```

---

### Multi-Agent in OpenAI (Responses API)

```json
{
  "model": "gpt-5",
  "instructions": "You are an orchestrator. Delegate tasks to specialist functions.",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "call_specialist",
        "description": "Calls a specialized AI agent for a focused task",
        "parameters": {
          "type": "object",
          "properties": {
            "agent_type": {
              "type": "string",
              "enum": ["code_reviewer", "test_writer", "doc_writer"]
            },
            "task": { "type": "string" }
          },
          "required": ["agent_type", "task"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "input": [
    { "role": "user", "content": "Build a complete implementation + tests + docs for a binary search function." }
  ]
}
```

---

## 13. FORMAT CONVERSION CHEAT SHEET

### Converting Anthropic → OpenAI

```python
def anthropic_to_openai(anthropic_request: dict) -> dict:
    """Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions"""
    messages = []
    
    # System prompt: Anthropic top-level → OpenAI first message
    if "system" in anthropic_request:
        system = anthropic_request["system"]
        if isinstance(system, list):
            system_text = " ".join(block["text"] for block in system if block.get("type") == "text")
        else:
            system_text = system
        messages.append({"role": "system", "content": system_text})
    
    # Convert messages
    for msg in anthropic_request.get("messages", []):
        role = msg["role"]  # user / assistant (same in both)
        content = msg["content"]
        
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            openai_content = []
            for block in content:
                if block["type"] == "text":
                    openai_content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    src = block["source"]
                    if src["type"] == "base64":
                        url = f"data:{src['media_type']};base64,{src['data']}"
                    else:
                        url = src["url"]
                    openai_content.append({"type": "image_url", "image_url": {"url": url}})
                elif block["type"] == "tool_use":
                    # Assistant tool call → OpenAI format
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"])
                            }
                        }]
                    })
                    openai_content = None
                    break
                elif block["type"] == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"] if isinstance(block["content"], str) 
                                   else json.dumps(block["content"])
                    })
                    openai_content = None
                    break
            if openai_content:
                messages.append({"role": role, "content": openai_content})
    
    # Convert tools
    openai_tools = []
    for tool in anthropic_request.get("tools", []):
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"]  # Same JSON Schema format
            }
        })
    
    result = {
        "model": anthropic_request.get("model", "gpt-4o"),
        "messages": messages,
        "max_tokens": anthropic_request.get("max_tokens", 1024)
    }
    if openai_tools:
        result["tools"] = openai_tools
    
    return result
```

---

### Converting OpenAI → Anthropic

```python
def openai_to_anthropic(openai_request: dict) -> dict:
    """Convert OpenAI /v1/chat/completions request to Anthropic /v1/messages"""
    system = None
    messages = []
    
    for msg in openai_request.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        
        if role in ("system", "developer"):
            # System → Anthropic top-level system
            system = content if isinstance(content, str) else content[0]["text"]
            
        elif role in ("user", "assistant"):
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                ant_content = []
                for block in content:
                    if block["type"] == "text":
                        ant_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            # Parse base64 data URL
                            header, data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            ant_content.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": data}
                            })
                        else:
                            ant_content.append({
                                "type": "image",
                                "source": {"type": "url", "url": url}
                            })
                
                # Handle tool_calls on assistant messages
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        ant_content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"])
                        })
                
                messages.append({"role": role, "content": ant_content})
                    
        elif role == "tool":
            # Tool results go in user message as tool_result blocks
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"]
                }]
            })
    
    # Convert tools
    ant_tools = []
    for tool in openai_request.get("tools", []):
        if tool["type"] == "function":
            fn = tool["function"]
            ant_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
            })
    
    result = {
        "model": openai_request.get("model", "claude-opus-4-6"),
        "max_tokens": openai_request.get("max_tokens", 1024),
        "messages": messages
    }
    if system:
        result["system"] = system
    if ant_tools:
        result["tools"] = ant_tools
    
    return result
```

---

### Converting OpenAI → Gemini

```python
def openai_to_gemini(openai_request: dict) -> dict:
    """Convert OpenAI /v1/chat/completions request to Gemini generateContent"""
    system_instruction = None
    contents = []
    
    for msg in openai_request.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        
        if role in ("system", "developer"):
            system_instruction = {
                "parts": [{"text": content if isinstance(content, str) else content[0]["text"]}]
            }
        elif role == "user":
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for block in content:
                    if block["type"] == "text":
                        parts.append({"text": block["text"]})
                    elif block["type"] == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            header, data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append({"inline_data": {"mime_type": mime, "data": data}})
                        else:
                            parts.append({"file_data": {"mime_type": "image/jpeg", "file_uri": url}})
            contents.append({"role": "user", "parts": parts})
        
        elif role == "assistant":
            parts = []
            if isinstance(content, str) and content:
                parts.append({"text": content})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    parts.append({
                        "functionCall": {
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"])
                        }
                    })
            if parts:
                contents.append({"role": "model", "parts": parts})
        
        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": msg.get("name", "tool"),
                        "response": {"content": msg["content"]}
                    }
                }]
            })
    
    # Convert tools
    gemini_tools = []
    fn_declarations = []
    for tool in openai_request.get("tools", []):
        if tool["type"] == "function":
            fn = tool["function"]
            fn_declarations.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {})
            })
    if fn_declarations:
        gemini_tools.append({"functionDeclarations": fn_declarations})
    
    result = {"contents": contents}
    if system_instruction:
        result["system_instruction"] = system_instruction
    if gemini_tools:
        result["tools"] = gemini_tools
    
    return result
```

---

## 14. RESPONSE ANATOMY

### Anthropic Response — All Stop Reasons

```json
{
  "id": "msg_01XkAbCdEf",
  "type": "message",
  "role": "assistant",
  "content": [
    { "type": "text", "text": "Hello!" }
    // OR:
    // { "type": "tool_use", "id": "toolu_01", "name": "fn_name", "input": {...} }
    // { "type": "thinking", "thinking": "...", "signature": "..." }
    // { "type": "redacted_thinking", "data": "..." }
  ],
  "model": "claude-opus-4-6",
  "stop_reason": "end_turn",
  // stop_reason values:
  // "end_turn"       - natural completion
  // "max_tokens"     - hit max_tokens limit
  // "stop_sequence"  - hit a stop sequence
  // "tool_use"       - model is calling a tool
  "stop_sequence": null,  // or the matched stop sequence string
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

---

### OpenAI Chat Completion Response — All Finish Reasons

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "gpt-4o-2024-11-20",
  "system_fingerprint": "fp_abc123",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!",
        // OR content: null when tool_calls present
        "tool_calls": null,  // or [{id, type, function: {name, arguments}}]
        "refusal": null      // or refusal message string
      },
      "finish_reason": "stop"
      // finish_reason values:
      // "stop"          - natural completion
      // "length"        - hit max_tokens
      // "tool_calls"    - model called tools
      // "content_filter"- content filtered
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_details": { "cached_tokens": 0, "audio_tokens": 0 },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  }
}
```

---

### Google Gemini Response — All Finish Reasons

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          { "text": "Hello!" }
          // OR: { "functionCall": { "name": "...", "args": {...} } }
          // OR: { "thought": true, "text": "thinking..." }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      // finishReason values:
      // "STOP"                 - natural completion
      // "MAX_TOKENS"           - hit token limit
      // "SAFETY"               - safety filter triggered
      // "RECITATION"           - recitation policy
      // "LANGUAGE"             - unsupported language
      // "FINISH_REASON_UNSPECIFIED" - default/unspecified
      "index": 0,
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "probability": "NEGLIGIBLE"
        }
      ]
    }
  ],
  "promptFeedback": {
    "safetyRatings": [...]
  },
  "usageMetadata": {
    "promptTokenCount": 100,
    "candidatesTokenCount": 50,
    "totalTokenCount": 150,
    "thoughtsTokenCount": 0
  }
}
```

---

## 15. ERROR FORMATS

### Anthropic Error

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens must be greater than 0"
  }
}
```

**Error types:** `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `rate_limit_error`, `api_error`, `overloaded_error`

---

### OpenAI Error

```json
{
  "error": {
    "message": "Invalid API Key provided",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

---

### Gemini Error

```json
{
  "error": {
    "code": 400,
    "message": "API key not valid. Please pass a valid API key.",
    "status": "INVALID_ARGUMENT",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID"
      }
    ]
  }
}
```

---

## 16. HEADERS & AUTH QUICK REFERENCE

### Anthropic Required Headers

```
x-api-key: sk-ant-api03-...
anthropic-version: 2023-06-01
content-type: application/json
```

**Optional beta headers:**
```
anthropic-beta: interleaved-thinking-2025-05-14
anthropic-beta: structured-outputs-2025-11-13
anthropic-beta: prompt-caching-2024-07-31
anthropic-beta: pdfs-2024-09-25
anthropic-beta: advanced-tool-use-2025-11-20
```

Multiple betas can be sent as comma-separated or multiple headers.

---

### OpenAI Required Headers

```
Authorization: Bearer sk-proj-...
content-type: application/json
```

---

### Gemini Required

**Google AI Studio (API key):**
```
x-goog-api-key: AIza...
content-type: application/json
```

**Vertex AI (OAuth):**
```
Authorization: Bearer $(gcloud auth print-access-token)
content-type: application/json
```

---

### Python `requests` Library Examples

**Anthropic:**
```python
import requests

response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": "sk-ant-...",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    },
    json={
        "model": "claude-opus-4-6",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
data = response.json()
text = data["content"][0]["text"]
```

**OpenAI:**
```python
import requests

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": "Bearer sk-proj-...",
        "content-type": "application/json"
    },
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
data = response.json()
text = data["choices"][0]["message"]["content"]
```

**Gemini:**
```python
import requests

response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    headers={
        "x-goog-api-key": "AIza...",
        "content-type": "application/json"
    },
    json={
        "contents": [{"role": "user", "parts": [{"text": "Hello!"}]}]
    }
)
data = response.json()
text = data["candidates"][0]["content"]["parts"][0]["text"]
```

---

## APPENDIX: CURRENT MODELS (Feb 2026)

| Provider | Latest Models | Notes |
|---|---|---|
| **Anthropic** | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 | Opus 4.6 has adaptive thinking; Sonnet 4.6 for balanced tasks |
| **OpenAI** | gpt-5, gpt-5.2, o3, o4-mini, gpt-4o | gpt-5 for most tasks; o3 for complex reasoning |
| **Google** | gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-2.5-flash | Gemini 3 uses thinking_level; 2.5 uses thinking_budget |

---

## APPENDIX: ROLE NAME MAPPING

| Concept | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Human turn | `user` | `user` | `user` |
| AI turn | `assistant` | `assistant` | `model` |
| System/Instructions | `system` (top-level) | `system` or `developer` (in messages) | `system_instruction` (top-level) |
| Tool result sender | `user` | `tool` | `user` |

---

*Last updated: February 2026 | Sources: docs.claude.com, platform.openai.com, ai.google.dev*

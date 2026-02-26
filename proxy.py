#!/usr/bin/env python3
"""
Universal LLM Proxy
===================
Converts between OpenAI / Anthropic / Gemini formats automatically.
Handles multiple providers, key rotation, failover, model mapping.

Usage:
  python proxy.py                          # auto-detect, uses default provider from config
  python proxy.py --to openrouter         # force target provider
  python proxy.py --from openai --to anthropic  # explicit format conversion
  python proxy.py --port 9000 --config custom.json
  python proxy.py --to groq --from auto   # auto-detect client format, send to groq
"""

import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────
# CLI ARGS
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Universal LLM Format-Converting Proxy")
    p.add_argument("--from", dest="from_format", default="auto",
                   choices=["auto", "openai", "anthropic", "gemini"],
                   help="Client request format (default: auto-detect)")
    p.add_argument("--to", dest="to_provider", default=None,
                   help="Target provider name from config (default: config default)")
    p.add_argument("--port", type=int, default=None, help="Port to listen on")
    p.add_argument("--host", default=None, help="Host to bind to")
    p.add_argument("--config", default="config.json", help="Config file path")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


# ─────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────

class Config:
    def __init__(self, path: str):
        with open(path) as f:
            raw = json.load(f)
        self.server = raw.get("server", {})
        self.defaults = raw.get("defaults", {})
        self.providers: Dict[str, dict] = {}
        for p in raw.get("providers", []):
            if p.get("enabled", True):
                self.providers[p["name"]] = p

    def get_provider(self, name: str) -> Optional[dict]:
        return self.providers.get(name)

    def default_provider_name(self) -> str:
        return self.defaults.get("to_provider", next(iter(self.providers)))

    def retry_codes(self) -> List[int]:
        return self.defaults.get("retry_on_codes", [429, 500, 502, 503, 504])

    def max_retries(self) -> int:
        return self.defaults.get("max_retries", 3)

    def timeout(self) -> int:
        return self.defaults.get("timeout_seconds", 120)


# ─────────────────────────────────────────────────────────
# KEY ROTATION STATE (per provider)
# ─────────────────────────────────────────────────────────

class KeyPool:
    def __init__(self, keys: List[str]):
        self._keys = list(keys)
        self._idx = 0
        self._lock = threading.Lock()
        self._cooldowns: Dict[str, float] = {}  # key → time when usable again

    def next(self) -> Optional[str]:
        with self._lock:
            now = time.time()
            total = len(self._keys)
            for _ in range(total):
                key = self._keys[self._idx]
                self._idx = (self._idx + 1) % total
                if now >= self._cooldowns.get(key, 0):
                    return key
            return None  # all on cooldown

    def mark_rate_limited(self, key: str, retry_after: int = 60):
        with self._lock:
            self._cooldowns[key] = time.time() + retry_after
            print(f"  ⏳ Key ...{key[-8:]} rate-limited for {retry_after}s")

    def mark_invalid(self, key: str):
        with self._lock:
            self._cooldowns[key] = time.time() + 3600  # 1 hour
            print(f"  ❌ Key ...{key[-8:]} marked invalid for 1h")


_key_pools: Dict[str, KeyPool] = {}

def get_key_pool(provider: dict) -> KeyPool:
    name = provider["name"]
    if name not in _key_pools:
        _key_pools[name] = KeyPool(provider.get("api_keys", []))
    return _key_pools[name]


# ─────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────

def detect_format(path: str, headers: dict, body: dict) -> str:
    """
    Detect the format of an incoming request.
    Returns: 'openai' | 'anthropic' | 'gemini'
    """
    path_l = path.lower()

    # Path-based detection
    if "/v1/messages" in path_l:
        return "anthropic"
    if "/v1beta/" in path_l or "/generatecontent" in path_l:
        return "gemini"
    if "/v1/chat/completions" in path_l or "/v1/completions" in path_l:
        return "openai"
    if "/v1/responses" in path_l:
        return "openai"

    # Header-based detection
    if "anthropic-version" in {k.lower() for k in headers}:
        return "anthropic"
    if "x-goog-api-key" in {k.lower() for k in headers}:
        return "gemini"

    # Body-based detection
    if body:
        if "contents" in body or "system_instruction" in body:
            return "gemini"
        if "anthropic_version" in body:
            return "anthropic"
        # Anthropic: top-level system (string/list) + messages (no system role in messages)
        if "messages" in body and "system" in body:
            msgs = body.get("messages", [])
            if msgs and msgs[0].get("role") not in ("system", "developer"):
                return "anthropic"

    return "openai"  # default


# ─────────────────────────────────────────────────────────
# MODEL MAPPING
# ─────────────────────────────────────────────────────────

def resolve_model(requested_model: str, provider: dict) -> str:
    """
    Map a requested model to one the provider supports.
    1. Check model_map for explicit mapping
    2. Check supported_models (if empty list → accept as-is)
    3. Fall back to primary_models[0]
    """
    model_map = provider.get("model_map", {})
    if requested_model in model_map:
        mapped = model_map[requested_model]
        if mapped != requested_model:
            print(f"  🔀 Model mapped: {requested_model} → {mapped}")
        return mapped

    supported = provider.get("supported_models", [])
    if not supported or requested_model in supported:
        return requested_model  # provider accepts it

    # Not in supported list → use primary
    primaries = provider.get("primary_models", [])
    if primaries:
        fallback = primaries[0]
        print(f"  🔀 Model '{requested_model}' not supported → fallback to '{fallback}'")
        return fallback

    return requested_model


# ─────────────────────────────────────────────────────────
# FORMAT CONVERTERS  (Request side)
# ─────────────────────────────────────────────────────────

class RequestConverter:

    @staticmethod
    def openai_to_anthropic(body: dict) -> dict:
        result = deepcopy(body)
        messages_in = result.pop("messages", [])
        system_parts = []
        messages_out = []

        for msg in messages_in:
            role = msg.get("role", "")
            content = msg.get("content")

            if role in ("system", "developer"):
                text = content if isinstance(content, str) else \
                    " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                system_parts.append({"type": "text", "text": text})

            elif role == "user":
                if isinstance(content, str):
                    messages_out.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    ant_content = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            ant_content.append({"type": "text", "text": block["text"]})
                        elif btype == "image_url":
                            url = block["image_url"]["url"]
                            if url.startswith("data:"):
                                header, data = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                ant_content.append({"type": "image", "source": {
                                    "type": "base64", "media_type": media_type, "data": data}})
                            else:
                                ant_content.append({"type": "image", "source": {
                                    "type": "url", "url": url}})
                    messages_out.append({"role": "user", "content": ant_content})

            elif role == "assistant":
                ant_content = []
                if content and isinstance(content, str):
                    ant_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            ant_content.append({"type": "text", "text": block["text"]})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc["function"]
                    ant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    })
                if ant_content:
                    messages_out.append({"role": "assistant", "content": ant_content})

            elif role == "tool":
                messages_out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "")
                }]})

        if system_parts:
            result["system"] = system_parts if len(system_parts) > 1 else system_parts[0]["text"]
        result["messages"] = messages_out

        # max_tokens rename (OpenAI uses max_tokens too, same field)
        if "max_completion_tokens" in result:
            result["max_tokens"] = result.pop("max_completion_tokens")
        if "max_tokens" not in result:
            result["max_tokens"] = 4096

        # Convert tools
        openai_tools = result.pop("tools", None)
        if openai_tools:
            ant_tools = []
            for tool in openai_tools:
                if tool.get("type") == "function":
                    fn = tool["function"]
                    ant_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
                    })
            result["tools"] = ant_tools

        # Strip OpenAI-only fields
        for field in ["response_format", "frequency_penalty", "presence_penalty",
                       "logit_bias", "user", "n", "reasoning_effort", "stream_options"]:
            result.pop(field, None)

        return result

    @staticmethod
    def anthropic_to_openai(body: dict) -> dict:
        result = deepcopy(body)
        messages_out = []

        system = result.pop("system", None)
        if system:
            text = system if isinstance(system, str) else \
                " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            messages_out.append({"role": "system", "content": text})

        for msg in result.pop("messages", []):
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                messages_out.append({"role": role, "content": content})
            elif isinstance(content, list):
                has_tool_use = any(b.get("type") == "tool_use" for b in content)
                has_tool_result = any(b.get("type") == "tool_result" for b in content)

                if has_tool_result:
                    for block in content:
                        if block.get("type") == "tool_result":
                            messages_out.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block.get("content", "")
                            })
                elif has_tool_use:
                    tool_calls = []
                    text_parts = []
                    for block in content:
                        if block["type"] == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"])
                                }
                            })
                        elif block["type"] == "text":
                            text_parts.append(block["text"])
                    messages_out.append({
                        "role": "assistant",
                        "content": " ".join(text_parts) or None,
                        "tool_calls": tool_calls
                    })
                else:
                    parts = []
                    for block in content:
                        if block.get("type") == "text":
                            parts.append({"type": "text", "text": block["text"]})
                        elif block.get("type") == "image":
                            src = block["source"]
                            if src["type"] == "base64":
                                url = f"data:{src['media_type']};base64,{src['data']}"
                            else:
                                url = src["url"]
                            parts.append({"type": "image_url", "image_url": {"url": url}})
                    if len(parts) == 1 and parts[0]["type"] == "text":
                        messages_out.append({"role": role, "content": parts[0]["text"]})
                    else:
                        messages_out.append({"role": role, "content": parts})

        result["messages"] = messages_out

        # Convert tools
        ant_tools = result.pop("tools", None)
        if ant_tools:
            oai_tools = []
            for tool in ant_tools:
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                    }
                })
            result["tools"] = oai_tools

        # Strip anthropic-only fields
        for field in ["thinking", "betas", "metadata"]:
            result.pop(field, None)

        return result

    @staticmethod
    def openai_to_gemini(body: dict) -> Tuple[dict, str]:
        """Returns (gemini_body, model_name)"""
        result = {}
        model = body.get("model", "gemini-2.0-flash")
        contents = []
        system_instruction = None

        for msg in body.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content")

            if role in ("system", "developer"):
                text = content if isinstance(content, str) else \
                    " ".join(b.get("text","") for b in content if isinstance(b,dict))
                system_instruction = {"parts": [{"text": text}]}

            elif role == "user":
                parts = []
                if isinstance(content, str):
                    parts = [{"text": content}]
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                        elif block.get("type") == "image_url":
                            url = block["image_url"]["url"]
                            if url.startswith("data:"):
                                header, data = url.split(",", 1)
                                mime = header.split(":")[1].split(";")[0]
                                parts.append({"inline_data": {"mime_type": mime, "data": data}})
                            else:
                                parts.append({"file_data": {"file_uri": url,
                                    "mime_type": "image/jpeg"}})
                contents.append({"role": "user", "parts": parts})

            elif role == "assistant":
                parts = []
                if content and isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc["function"]
                    parts.append({"functionCall": {
                        "name": fn["name"],
                        "args": json.loads(fn["arguments"]) if isinstance(fn["arguments"],str) else fn["arguments"]
                    }})
                if parts:
                    contents.append({"role": "model", "parts": parts})

            elif role == "tool":
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {
                        "name": msg.get("name", "tool"),
                        "response": {"content": msg.get("content", "")}
                    }
                }]})

        result["contents"] = contents
        if system_instruction:
            result["system_instruction"] = system_instruction

        # Tools
        oai_tools = body.get("tools", [])
        if oai_tools:
            fn_decls = []
            for tool in oai_tools:
                if tool.get("type") == "function":
                    fn = tool["function"]
                    fn_decls.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {})
                    })
            result["tools"] = [{"functionDeclarations": fn_decls}]

        # generation_config
        gen_cfg = {}
        if "max_tokens" in body: gen_cfg["maxOutputTokens"] = body["max_tokens"]
        if "max_completion_tokens" in body: gen_cfg["maxOutputTokens"] = body["max_completion_tokens"]
        if "temperature" in body: gen_cfg["temperature"] = body["temperature"]
        if "top_p" in body: gen_cfg["topP"] = body["top_p"]
        if "stop" in body:
            stop = body["stop"]
            gen_cfg["stopSequences"] = [stop] if isinstance(stop, str) else stop
        if gen_cfg:
            result["generation_config"] = gen_cfg

        return result, model

    @staticmethod
    def gemini_to_openai(body: dict) -> dict:
        result = {}
        messages = []

        si = body.get("system_instruction")
        if si:
            text = " ".join(p.get("text","") for p in si.get("parts",[]))
            messages.append({"role": "system", "content": text})

        for c in body.get("contents", []):
            role = c.get("role","user")
            oai_role = "assistant" if role == "model" else "user"
            parts = c.get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]
            fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            fn_responses = [p["functionResponse"] for p in parts if "functionResponse" in p]

            if fn_responses:
                for fr in fn_responses:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": fr["name"],
                        "name": fr["name"],
                        "content": json.dumps(fr.get("response", {}))
                    })
            elif fn_calls:
                tool_calls = [{"id": fc["name"], "type": "function", "function": {
                    "name": fc["name"], "arguments": json.dumps(fc.get("args", {}))
                }} for fc in fn_calls]
                messages.append({
                    "role": oai_role,
                    "content": " ".join(text_parts) or None,
                    "tool_calls": tool_calls
                })
            else:
                content = " ".join(text_parts)
                messages.append({"role": oai_role, "content": content})

        result["messages"] = messages

        # Tools back
        for tool_block in body.get("tools", []):
            fn_decls = tool_block.get("functionDeclarations", [])
            if fn_decls:
                result["tools"] = [{"type": "function", "function": {
                    "name": fd["name"],
                    "description": fd.get("description",""),
                    "parameters": fd.get("parameters", {})
                }} for fd in fn_decls]

        # generation_config back
        gc = body.get("generation_config", {})
        if "maxOutputTokens" in gc: result["max_tokens"] = gc["maxOutputTokens"]
        if "temperature" in gc: result["temperature"] = gc["temperature"]
        if "topP" in gc: result["top_p"] = gc["topP"]

        return result

    @staticmethod
    def anthropic_to_gemini(body: dict) -> Tuple[dict, str]:
        openai_body = RequestConverter.anthropic_to_openai(body)
        return RequestConverter.openai_to_gemini(openai_body)

    @staticmethod
    def gemini_to_anthropic(body: dict) -> dict:
        openai_body = RequestConverter.gemini_to_openai(body)
        return RequestConverter.openai_to_anthropic(openai_body)


def convert_request(body: dict, from_fmt: str, to_fmt: str) -> Tuple[dict, Optional[str]]:
    """
    Convert request body from from_fmt to to_fmt.
    Returns (converted_body, model_name_if_gemini)
    """
    if from_fmt == to_fmt:
        if to_fmt == "gemini":
            return body, body.get("model", "gemini-2.0-flash")
        return body, None

    if from_fmt == "openai" and to_fmt == "anthropic":
        return RequestConverter.openai_to_anthropic(body), None
    if from_fmt == "anthropic" and to_fmt == "openai":
        return RequestConverter.anthropic_to_openai(body), None
    if from_fmt == "openai" and to_fmt == "gemini":
        result, model = RequestConverter.openai_to_gemini(body)
        return result, model
    if from_fmt == "gemini" and to_fmt == "openai":
        return RequestConverter.gemini_to_openai(body), None
    if from_fmt == "anthropic" and to_fmt == "gemini":
        result, model = RequestConverter.anthropic_to_gemini(body)
        return result, model
    if from_fmt == "gemini" and to_fmt == "anthropic":
        return RequestConverter.gemini_to_anthropic(body), None

    return body, None


# ─────────────────────────────────────────────────────────
# FORMAT CONVERTERS  (Response side)
# ─────────────────────────────────────────────────────────

class ResponseConverter:
    """Convert provider responses back to the client's expected format."""

    @staticmethod
    def _gemini_to_openai_response(gemini_resp: dict, model: str) -> dict:
        candidates = gemini_resp.get("candidates", [])
        choices = []
        for i, cand in enumerate(candidates):
            parts = cand.get("content", {}).get("parts", [])
            text_parts = [p.get("text","") for p in parts if "text" in p and not p.get("thought")]
            fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if fn_calls:
                tool_calls = [{"id": fc["name"], "type": "function", "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("args", {}))
                }} for fc in fn_calls]
                msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "".join(text_parts)}
                finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
                finish = finish_map.get(cand.get("finishReason","STOP"), "stop")

            choices.append({"index": i, "message": msg, "finish_reason": finish})

        usage = gemini_resp.get("usageMetadata", {})
        return {
            "id": f"chatcmpl-gemini-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
        }

    @staticmethod
    def _openai_to_anthropic_response(oai_resp: dict) -> dict:
        choices = oai_resp.get("choices", [])
        content = []
        stop_reason = "end_turn"
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content")
            if text:
                content.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc["function"]
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": json.loads(fn["arguments"]) if isinstance(fn["arguments"],str) else fn["arguments"]
                })
            finish = choices[0].get("finish_reason","stop")
            stop_reason = {"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use"}.get(finish,"end_turn")

        usage = oai_resp.get("usage", {})
        return {
            "id": f"msg_{oai_resp.get('id','')}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": oai_resp.get("model",""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }
        }

    @staticmethod
    def _anthropic_to_openai_response(ant_resp: dict) -> dict:
        content_blocks = ant_resp.get("content", [])
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        finish_map = {"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls","stop_sequence":"stop"}
        finish = finish_map.get(ant_resp.get("stop_reason","end_turn"),"stop")

        if tool_uses:
            tool_calls = [{"id": tu["id"], "type": "function", "function": {
                "name": tu["name"],
                "arguments": json.dumps(tu["input"])
            }} for tu in tool_uses]
            msg = {"role": "assistant", "content": "".join(text_parts) or None, "tool_calls": tool_calls}
        else:
            msg = {"role": "assistant", "content": "".join(text_parts), "refusal": None}

        usage = ant_resp.get("usage", {})
        return {
            "id": f"chatcmpl-{ant_resp.get('id','')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ant_resp.get("model",""),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens",0) + usage.get("output_tokens",0)
            }
        }

    @staticmethod
    def _openai_to_gemini_response(oai_resp: dict) -> dict:
        choices = oai_resp.get("choices", [])
        candidates = []
        for i, ch in enumerate(choices):
            msg = ch.get("message", {})
            parts = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg.get("tool_calls",[]) or []:
                fn = tc["function"]
                parts.append({"functionCall": {
                    "name": fn["name"],
                    "args": json.loads(fn["arguments"]) if isinstance(fn["arguments"],str) else fn["arguments"]
                }})
            finish_map = {"stop":"STOP","length":"MAX_TOKENS","tool_calls":"STOP","content_filter":"SAFETY"}
            candidates.append({
                "content": {"parts": parts, "role": "model"},
                "finishReason": finish_map.get(ch.get("finish_reason","stop"),"STOP"),
                "index": i
            })
        usage = oai_resp.get("usage",{})
        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens",0),
                "candidatesTokenCount": usage.get("completion_tokens",0),
                "totalTokenCount": usage.get("total_tokens",0)
            }
        }

    @staticmethod
    def _anthropic_to_gemini_response(ant_resp: dict) -> dict:
        oai = ResponseConverter._anthropic_to_openai_response(ant_resp)
        return ResponseConverter._openai_to_gemini_response(oai)

    @staticmethod
    def _gemini_to_anthropic_response(gemini_resp: dict, model: str) -> dict:
        oai = ResponseConverter._gemini_to_openai_response(gemini_resp, model)
        return ResponseConverter._openai_to_anthropic_response(oai)

    @staticmethod
    def convert(resp_body: dict, provider_fmt: str, client_fmt: str, model: str = "") -> dict:
        if provider_fmt == client_fmt:
            return resp_body

        # Normalize to openai as intermediate
        if provider_fmt == "gemini":
            oai = ResponseConverter._gemini_to_openai_response(resp_body, model)
        elif provider_fmt == "anthropic":
            oai = ResponseConverter._anthropic_to_openai_response(resp_body)
        else:
            oai = resp_body

        if client_fmt == "openai":
            return oai
        elif client_fmt == "anthropic":
            return ResponseConverter._openai_to_anthropic_response(oai)
        elif client_fmt == "gemini":
            return ResponseConverter._openai_to_gemini_response(oai)

        return resp_body


# ─────────────────────────────────────────────────────────
# STREAMING CONVERTERS
# ─────────────────────────────────────────────────────────

class StreamConverter:
    """
    Convert SSE streams between formats.
    Strategy:
      - Same format → passthrough
      - Different format → buffer full stream, convert, re-emit
        (most reliable; avoids partial-chunk edge cases)
    """

    @staticmethod
    def openai_chunk_to_anthropic(chunk_data: dict, index: int = 0) -> List[str]:
        """Convert one OpenAI SSE chunk → list of Anthropic SSE events."""
        choices = chunk_data.get("choices", [])
        events = []
        if not choices:
            return events
        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        if delta.get("content"):
            events.append(f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':delta['content']}})}\n\n")
        if finish:
            stop_map = {"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use"}
            events.append(f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_map.get(finish,'end_turn')}})}\n\n")
            events.append(f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n")
        return events

    @staticmethod
    def anthropic_events_to_openai(events: List[Tuple[str, dict]]) -> List[str]:
        """Convert list of (event_name, data_dict) → OpenAI SSE chunks."""
        chunks = []
        msg_id = f"chatcmpl-ant-{int(time.time())}"
        for ev_name, data in events:
            if ev_name == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = {"id": msg_id, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {"content": delta["text"]}, "finish_reason": None}]}
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")
            elif ev_name == "message_delta":
                stop_reason = data.get("delta", {}).get("stop_reason", "end_turn")
                finish_map = {"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls"}
                chunk = {"id": msg_id, "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish_map.get(stop_reason,"stop")}]}
                chunks.append(f"data: {json.dumps(chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return chunks

    @staticmethod
    def gemini_chunks_to_openai(gemini_chunks: List[dict], model: str) -> List[str]:
        """Convert Gemini stream chunks → OpenAI SSE chunks."""
        chunks = []
        msg_id = f"chatcmpl-gem-{int(time.time())}"
        for gc in gemini_chunks:
            for cand in gc.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if "text" in part:
                        chunk = {"id": msg_id, "object": "chat.completion.chunk", "model": model,
                                 "choices": [{"index": 0, "delta": {"content": part["text"]}, "finish_reason": None}]}
                        chunks.append(f"data: {json.dumps(chunk)}\n\n")
                finish = cand.get("finishReason","")
                if finish == "STOP":
                    chunk = {"id": msg_id, "object": "chat.completion.chunk", "model": model,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    chunks.append(f"data: {json.dumps(chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return chunks


def build_anthropic_stream_preamble(model: str, input_tokens: int = 0) -> List[str]:
    """Build the Anthropic SSE preamble events."""
    msg_id = f"msg_proxy_{int(time.time())}"
    events = [
        f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'usage':{'input_tokens':input_tokens,'output_tokens':0}}})}\n\n",
        f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n",
        f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"
    ]
    return events


# ─────────────────────────────────────────────────────────
# URL BUILDER
# ─────────────────────────────────────────────────────────

def build_target_url(provider: dict, path: str, client_path: str,
                     client_fmt: str, model: str = "") -> str:
    base = provider["base_url"].rstrip("/")
    native_fmt = provider["native_format"]

    if native_fmt == "gemini":
        # Gemini needs model in URL
        is_stream = "stream=true" in client_path.lower() or path.endswith("stream")
        action = "streamGenerateContent" if is_stream else "generateContent"
        m = model or "gemini-2.0-flash"
        return f"{base}/v1beta/models/{m}:{action}"

    if native_fmt == "anthropic":
        return f"{base}/v1/messages"

    # openai-compatible: preserve the client's path if it makes sense
    if "/v1/chat/completions" in client_path:
        return f"{base}/v1/chat/completions"
    if "/api/chat" in client_path:
        return f"{base}/api/chat"
    if "/v1/models" in client_path:
        return f"{base}/v1/models"
    # fallback
    return f"{base}/v1/chat/completions"


# ─────────────────────────────────────────────────────────
# HTTP PROXY HANDLER
# ─────────────────────────────────────────────────────────

_config: Optional[Config] = None
_args = None
_verbose = False

class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default

    def _log(self, msg: str):
        print(msg, flush=True)

    def do_GET(self):
        # Pass through model listing etc.
        self._handle_passthrough()

    def do_DELETE(self):
        self._handle_passthrough()

    def do_POST(self):
        self._handle_llm()

    def _handle_passthrough(self):
        """For non-LLM paths (GET /v1/models, etc.) just proxy to the provider."""
        provider_name = _args.to_provider or _config.default_provider_name()
        provider = _config.get_provider(provider_name)
        if not provider:
            self._send_error(400, f"Provider '{provider_name}' not found or not enabled")
            return

        pool = get_key_pool(provider)
        key = pool.next()
        target_url = build_target_url(provider, self.path, self.path, "openai")
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host","authorization")}
        headers["Host"] = urllib.parse.urlparse(provider["base_url"]).netloc
        headers.update(provider.get("extra_headers", {}))
        if key:
            if provider["native_format"] == "anthropic":
                headers["x-api-key"] = key
            elif provider["native_format"] == "gemini":
                headers["x-goog-api-key"] = key
            else:
                headers["Authorization"] = f"Bearer {key}"

        try:
            req = urllib.request.Request(target_url, headers=headers, method=self.command)
            with urllib.request.urlopen(req, timeout=_config.timeout()) as resp:
                self.send_response(resp.getcode())
                for h, v in resp.getheaders():
                    if h.lower() not in ("connection","transfer-encoding"):
                        self.send_header(h, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self._send_error(502, str(e))

    def _handle_llm(self):
        ts = datetime.now().strftime("%H:%M:%S")

        # Read body
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw_body) if raw_body else {}
        except Exception as e:
            self._send_error(400, f"Invalid JSON body: {e}")
            return

        # Detect client format
        if _args.from_format == "auto":
            client_fmt = detect_format(self.path, dict(self.headers), body)
        else:
            client_fmt = _args.from_format

        # Determine target provider
        provider_name = _args.to_provider or _config.default_provider_name()
        provider = _config.get_provider(provider_name)
        if not provider:
            self._send_error(400, f"Provider '{provider_name}' not found or not enabled in config")
            return

        target_fmt = provider["native_format"]
        is_stream = body.get("stream", False)

        # Extract & resolve model
        orig_model = body.get("model", "")
        resolved_model = resolve_model(orig_model, provider)
        if orig_model != resolved_model:
            body = deepcopy(body)
            body["model"] = resolved_model

        self._log(f"\n[{ts}] POST {self.path}")
        self._log(f"  📥 Client format : {client_fmt.upper()}")
        self._log(f"  📤 Provider       : {provider_name} ({target_fmt.upper()})")
        self._log(f"  🤖 Model          : {orig_model or '(none)'}" +
                  (f" → {resolved_model}" if orig_model != resolved_model else ""))
        self._log(f"  🌊 Streaming      : {is_stream}")

        # Convert request
        try:
            converted_body, gemini_model = convert_request(body, client_fmt, target_fmt)
            gemini_model = gemini_model or resolved_model
        except Exception as e:
            self._send_error(500, f"Request conversion error: {e}\n{traceback.format_exc()}")
            return

        target_url = build_target_url(provider, self.path, self.path,
                                       client_fmt, gemini_model)

        # Execute with retry/failover
        self._execute_with_retry(
            provider=provider,
            converted_body=converted_body,
            target_url=target_url,
            client_fmt=client_fmt,
            target_fmt=target_fmt,
            is_stream=is_stream,
            model=gemini_model or resolved_model,
            orig_path=self.path
        )

    def _execute_with_retry(self, provider, converted_body, target_url,
                             client_fmt, target_fmt, is_stream, model, orig_path):
        pool = get_key_pool(provider)
        max_retries = _config.max_retries()
        retry_codes = _config.retry_codes()
        last_error = None

        for attempt in range(max_retries):
            key = pool.next()
            if key is None:
                self._send_error(429, "All API keys are on cooldown. Try again later.")
                return

            key_suffix = f"...{key[-8:]}"
            self._log(f"  🔑 Key #{attempt+1}: {key_suffix}")

            # Build headers
            headers = {"Content-Type": "application/json"}
            headers.update(provider.get("extra_headers", {}))
            headers["Host"] = urllib.parse.urlparse(provider["base_url"]).netloc

            if target_fmt == "anthropic":
                headers["x-api-key"] = key
                if "anthropic-version" not in {k.lower() for k in headers}:
                    headers["anthropic-version"] = "2023-06-01"
            elif target_fmt == "gemini":
                headers["x-goog-api-key"] = key
            else:
                headers["Authorization"] = f"Bearer {key}"

            payload = json.dumps(converted_body).encode("utf-8")
            req = urllib.request.Request(
                url=target_url,
                data=payload,
                headers=headers,
                method="POST"
            )

            try:
                resp = urllib.request.urlopen(req, timeout=_config.timeout())
                status = resp.getcode()
                self._log(f"  ✅ Response: {status}")
                self._forward_response(resp, client_fmt, target_fmt, is_stream, model)
                return

            except urllib.error.HTTPError as e:
                self._log(f"  ⚠️  HTTP {e.code} on attempt {attempt+1}")
                if e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", 60))
                    pool.mark_rate_limited(key, retry_after)
                elif e.code == 401:
                    pool.mark_invalid(key)
                elif e.code not in retry_codes:
                    # Not retryable — forward error as-is
                    self.send_response(e.code)
                    for h, v in e.headers.items():
                        if h.lower() not in ("connection","transfer-encoding"):
                            self.send_header(h, v)
                    self.end_headers()
                    self.wfile.write(e.read())
                    return
                last_error = e

            except Exception as e:
                self._log(f"  ⚠️  Error on attempt {attempt+1}: {e}")
                last_error = e

        # All retries exhausted
        self._log(f"  ❌ All {max_retries} attempts failed")
        self._send_error(502, f"All retry attempts failed. Last error: {last_error}")

    def _forward_response(self, resp, client_fmt: str, target_fmt: str,
                           is_stream: bool, model: str):
        """Read provider response, convert format, send to client."""

        if is_stream:
            self._forward_stream(resp, client_fmt, target_fmt, model)
        else:
            raw = resp.read()
            try:
                resp_body = json.loads(raw)
            except Exception:
                # Not JSON — passthrough
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
                return

            if client_fmt != target_fmt:
                try:
                    resp_body = ResponseConverter.convert(resp_body, target_fmt, client_fmt, model)
                except Exception as e:
                    self._log(f"  ⚠️  Response conversion error: {e}")

            out = json.dumps(resp_body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def _forward_stream(self, resp, client_fmt: str, target_fmt: str, model: str):
        """Handle SSE streaming with format conversion."""
        same_format = (client_fmt == target_fmt)

        if same_format:
            # Pure passthrough — relay chunks directly
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception:
                pass
            return

        # Different formats — buffer the upstream stream, then convert+emit
        # Collect all SSE data lines
        buffer = b""
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                buffer += chunk
        except Exception as e:
            self._log(f"  ⚠️  Stream read error: {e}")

        text = buffer.decode("utf-8", errors="replace")

        # Parse SSE events
        parsed_events = self._parse_sse(text)

        # Generate output SSE text
        output_sse = self._convert_stream(parsed_events, client_fmt, target_fmt, model)

        out_bytes = output_sse.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(out_bytes)))
        self.end_headers()
        self.wfile.write(out_bytes)
        self.wfile.flush()

    @staticmethod
    def _parse_sse(text: str) -> List[Tuple[str, Any]]:
        """
        Parse SSE stream into list of (event_name, data_object).
        Handles both named events (Anthropic style) and plain data: (OpenAI/Gemini style).
        """
        events = []
        current_event = "message"
        for line in text.splitlines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    events.append(("done", None))
                    current_event = "message"
                else:
                    try:
                        data = json.loads(data_str)
                        events.append((current_event, data))
                    except Exception:
                        pass
                    current_event = "message"
            elif line == "":
                current_event = "message"
        return events

    @staticmethod
    def _convert_stream(events: List[Tuple[str, Any]], client_fmt: str,
                         target_fmt: str, model: str) -> str:
        """Convert parsed SSE events to client format SSE string."""
        out = ""

        # Build full text + tool calls from events (format-agnostic reconstruction)
        full_text = ""
        tool_calls_acc = []
        stop_reason = "end_turn"

        for ev_name, data in events:
            if data is None:  # [DONE]
                continue

            if target_fmt == "openai":
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        full_text += delta["content"]
                    finish = choices[0].get("finish_reason")
                    if finish:
                        fmap = {"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use"}
                        stop_reason = fmap.get(finish,"end_turn")

            elif target_fmt == "anthropic":
                if ev_name == "content_block_delta":
                    delta = data.get("delta",{})
                    if delta.get("type") == "text_delta":
                        full_text += delta.get("text","")
                elif ev_name == "message_delta":
                    stop_reason = data.get("delta",{}).get("stop_reason","end_turn")

            elif target_fmt == "gemini":
                for cand in data.get("candidates",[]):
                    for part in cand.get("content",{}).get("parts",[]):
                        if "text" in part:
                            full_text += part["text"]
                    if cand.get("finishReason") == "STOP":
                        stop_reason = "end_turn"

        # Now emit in client format
        if client_fmt == "openai":
            msg_id = f"chatcmpl-proxy-{int(time.time())}"
            if full_text:
                chunk = {"id": msg_id, "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":None}]}
                out += f"data: {json.dumps(chunk)}\n\n"
                # Emit text in pieces (simulate streaming feel)
                chunk = {"id": msg_id, "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index":0,"delta":{"content":full_text},"finish_reason":None}]}
                out += f"data: {json.dumps(chunk)}\n\n"
            finish_oai = {"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls"}.get(stop_reason,"stop")
            chunk = {"id": msg_id, "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index":0,"delta":{},"finish_reason":finish_oai}]}
            out += f"data: {json.dumps(chunk)}\n\n"
            out += "data: [DONE]\n\n"

        elif client_fmt == "anthropic":
            msg_id = f"msg_proxy_{int(time.time())}"
            out += f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
            out += f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
            out += f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"
            if full_text:
                out += f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':full_text}})}\n\n"
            out += f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
            out += f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
            out += f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

        elif client_fmt == "gemini":
            out += f"data: {json.dumps({'candidates':[{'content':{'parts':[{'text':full_text}],'role':'model'},'finishReason':'STOP','index':0}]})}\n\n"

        return out

    def _send_error(self, code: int, msg: str):
        body = json.dumps({"error": {"message": msg, "type": "proxy_error", "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ─────────────────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────────────────

def main():
    global _config, _args, _verbose

    _args = parse_args()
    _verbose = _args.verbose

    # Load config
    config_path = _args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file '{config_path}' not found. Create it or use --config <path>")
        sys.exit(1)

    _config = Config(config_path)

    # Resolve server params
    host = _args.host or _config.server.get("host", "localhost")
    port = _args.port or _config.server.get("port", 8000)
    provider_name = _args.to_provider or _config.default_provider_name()

    if provider_name not in _config.providers:
        available = list(_config.providers.keys())
        print(f"❌ Provider '{provider_name}' not enabled. Available: {available}")
        sys.exit(1)

    provider = _config.providers[provider_name]
    native_fmt = provider["native_format"]

    print("\n" + "═"*60)
    print("  🔀  Universal LLM Proxy")
    print("═"*60)
    print(f"  Listening  : http://{host}:{port}")
    print(f"  From       : {_args.from_format.upper()} (auto-detect if 'auto')")
    print(f"  To         : {provider_name.upper()} [{native_fmt.upper()} format]")
    print(f"  Target     : {provider['base_url']}")
    print(f"  API Keys   : {len(provider.get('api_keys', []))} loaded")
    print(f"  Max Retry  : {_config.max_retries()}")
    print("═"*60)
    print("\n  Proxy path examples:")
    if native_fmt == "anthropic":
        endpoint = "/v1/messages"
    elif native_fmt == "gemini":
        endpoint = "/v1beta/models/<model>:generateContent"
    else:
        endpoint = "/v1/chat/completions"
    print(f"  http://{host}:{port}{endpoint}")
    print(f"\n  Or send ANY format to ANY path — auto-detection handles it.")
    print(f"\n  Client config example:")
    print(f"    base_url = 'http://{host}:{port}'")
    print(f"    api_key  = 'proxy-key'  # any string works")
    print("═"*60 + "\n")

    import socketserver
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, port), ProxyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  🛑 Proxy stopped.\n")


if __name__ == "__main__":
    main()

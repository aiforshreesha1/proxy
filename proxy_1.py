#!/usr/bin/env python3
"""
Universal LLM Proxy
===================
Auto-detects client request format (OpenAI / Anthropic / Gemini).
Converts to the target provider's native format. Converts response back.
Handles key rotation, rate-limit cooldown, model mapping, streaming, tool calling.

Usage:
  python proxy.py                              # use default provider from config
  python proxy.py --provider groq              # use groq
  python proxy.py --provider ollama            # use ollama
  python proxy.py --provider openrouter        # use openrouter
  python proxy.py --provider gemini --to openai   # send to gemini but respond in openai format
  python proxy.py --provider anthropic --to anthropic  # force anthropic<->anthropic (no convert)
  python proxy.py --from openai               # hint client format instead of auto-detect
  python proxy.py --port 9000 --config my.json

--provider : which backend to use  (matches "name" in config.json providers)
--from     : hint the client format if auto-detect is wrong (openai|anthropic|gemini)
--to       : override the response format sent back to client   (openai|anthropic|gemini)
             default: same as client's detected format
"""

import argparse
import copy
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"

# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal LLM Format-Converting Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python proxy.py --provider groq
  python proxy.py --provider ollama --to anthropic
  python proxy.py --provider openrouter --from anthropic
  python proxy.py --provider gemini --port 9000
"""
    )
    p.add_argument("--provider", default=None,
                   help="Backend provider name (from config). Default: config defaults.provider")
    p.add_argument("--from", dest="from_format", default="auto",
                   choices=["auto", "openai", "anthropic", "gemini"],
                   help="Client request format hint. Default: auto-detect")
    p.add_argument("--to", dest="to_format", default=None,
                   choices=["openai", "anthropic", "gemini"],
                   help="Force response format back to client. Default: mirrors client format")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--config", default="config.json")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

class Config:
    def __init__(self, path: str):
        if not os.path.exists(path):
            print(f"{RED}✗ Config '{path}' not found.{RESET}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.server   = raw.get("server", {})
        self.defaults = raw.get("defaults", {})
        self.providers: Dict[str, dict] = {}
        for prov in raw.get("providers", []):
            if prov.get("enabled", True):
                self.providers[prov["name"]] = prov

    def get(self, name: str) -> Optional[dict]:
        return self.providers.get(name)

    @property
    def default_provider(self) -> str:
        return self.defaults.get("provider", next(iter(self.providers), ""))

    @property
    def retry_codes(self) -> List[int]:
        return self.defaults.get("retry_on_codes", [429, 500, 502, 503, 504])

    @property
    def max_retries(self) -> int:
        return self.defaults.get("max_retries", 3)

    @property
    def timeout(self) -> int:
        return self.defaults.get("timeout_seconds", 120)


# ─────────────────────────────────────────────────────────────────
# KEY POOL  (per-provider, round-robin with cooldown)
# ─────────────────────────────────────────────────────────────────

class KeyPool:
    def __init__(self, keys: List[str]):
        self._keys = list(keys)
        self._idx  = 0
        self._lock = threading.Lock()
        self._cooldown: Dict[str, float] = {}

    def next_key(self) -> Optional[str]:
        with self._lock:
            now = time.time()
            for _ in range(len(self._keys)):
                key = self._keys[self._idx]
                self._idx = (self._idx + 1) % len(self._keys)
                if now >= self._cooldown.get(key, 0):
                    return key
        return None  # all on cooldown

    def rate_limit(self, key: str, retry_after: int = 60):
        with self._lock:
            self._cooldown[key] = time.time() + retry_after
        log(f"  ⏳ Key …{key[-8:]} rate-limited for {retry_after}s", YELLOW)

    def invalidate(self, key: str, seconds: int = 3600):
        with self._lock:
            self._cooldown[key] = time.time() + seconds
        log(f"  ❌ Key …{key[-8:]} marked invalid for {seconds//60}m", RED)


_key_pools: Dict[str, KeyPool] = {}
_kp_lock = threading.Lock()

def get_pool(provider: dict) -> KeyPool:
    name = provider["name"]
    with _kp_lock:
        if name not in _key_pools:
            _key_pools[name] = KeyPool(provider.get("api_keys", []))
    return _key_pools[name]


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────

_verbose = False

def log(msg: str, color: str = ""):
    print(f"{color}{msg}{RESET}" if color else msg, flush=True)

def vlog(msg: str, color: str = GREY):
    if _verbose:
        print(f"{color}{msg}{RESET}", flush=True)


# ─────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_client_format(path: str, headers: Dict[str, str], body: dict) -> str:
    """Return 'openai' | 'anthropic' | 'gemini' from the incoming request."""
    pl = path.lower()

    # Path clues
    if "/v1/messages" in pl:
        return "anthropic"
    if "/v1beta/" in pl or "generatecontent" in pl:
        return "gemini"
    if "/v1/chat/completions" in pl or "/v1/completions" in pl or "/v1/responses" in pl:
        return "openai"
    if "/api/chat" in pl or "/api/generate" in pl:
        return "openai"  # ollama-local style

    # Header clues
    lh = {k.lower() for k in headers}
    if "anthropic-version" in lh or "x-api-key" in lh:
        return "anthropic"
    if "x-goog-api-key" in lh:
        return "gemini"

    # Body clues
    if "contents" in body or "system_instruction" in body:
        return "gemini"
    if "system" in body and "messages" in body:
        msgs = body.get("messages", [])
        if msgs and msgs[0].get("role") not in ("system", "developer"):
            return "anthropic"
    if "thinking" in body or "anthropic_version" in body:
        return "anthropic"

    return "openai"


# ─────────────────────────────────────────────────────────────────
# MODEL RESOLUTION
# ─────────────────────────────────────────────────────────────────

def resolve_model(requested: str, provider: dict) -> str:
    """
    1. Check model_map (optional key — may be absent or empty)
    2. Check supported_models whitelist (empty = accept anything)
    3. Fall back to primary_models[0]
    """
    model_map = provider.get("model_map") or {}
    if requested in model_map:
        mapped = model_map[requested]
        if mapped != requested:
            log(f"  🔀 model_map: {CYAN}{requested}{RESET} → {GREEN}{mapped}{RESET}")
        return mapped

    supported = provider.get("supported_models") or []
    if not supported or requested in supported:
        return requested  # provider accepts it

    # Not in whitelist → primary fallback
    primaries = provider.get("primary_models") or []
    if primaries:
        fb = primaries[0]
        log(f"  🔀 fallback:  {CYAN}{requested}{RESET} → {YELLOW}{fb}{RESET} (not in supported_models)")
        return fb

    return requested


# ─────────────────────────────────────────────────────────────────
# REQUEST CONVERTERS
# ─────────────────────────────────────────────────────────────────

class ReqConv:
    """Convert a request body from one format to another."""

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _oai_content_to_anthropic_parts(content) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        parts = []
        for block in content:
            t = block.get("type")
            if t == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif t == "image_url":
                url = block["image_url"]["url"]
                if url.startswith("data:"):
                    header, data = url.split(",", 1)
                    mt = header.split(":")[1].split(";")[0]
                    parts.append({"type": "image",
                                  "source": {"type": "base64", "media_type": mt, "data": data}})
                else:
                    parts.append({"type": "image",
                                  "source": {"type": "url", "url": url}})
            elif t == "input_audio":
                # Responses API audio — skip silently
                pass
        return parts

    @staticmethod
    def _ant_content_to_oai_parts(blocks) -> List[dict]:
        if isinstance(blocks, str):
            return [{"type": "text", "text": blocks}]
        parts = []
        for b in blocks:
            bt = b.get("type")
            if bt == "text":
                parts.append({"type": "text", "text": b["text"]})
            elif bt == "image":
                src = b["source"]
                if src["type"] == "base64":
                    url = f"data:{src['media_type']};base64,{src['data']}"
                else:
                    url = src["url"]
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif bt == "thinking":
                # surface as text comment for non-anthropic clients
                parts.append({"type": "text",
                               "text": f"<thinking>{b.get('thinking','')}</thinking>"})
        return parts

    @staticmethod
    def _parse_args(raw) -> dict:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return raw or {}

    # ── OpenAI → Anthropic ───────────────────────────────────────

    @staticmethod
    def openai_to_anthropic(body: dict) -> dict:
        out: dict = {}
        messages_in = body.get("messages", [])
        system_parts: List[dict] = []
        messages_out: List[dict] = []

        for msg in messages_in:
            role    = msg.get("role", "")
            content = msg.get("content")

            if role in ("system", "developer"):
                text = content if isinstance(content, str) \
                    else " ".join(b.get("text","") for b in (content or []) if isinstance(b,dict))
                system_parts.append({"type": "text", "text": text})

            elif role == "user":
                ant_content = ReqConv._oai_content_to_anthropic_parts(content or "")
                messages_out.append({"role": "user", "content": ant_content})

            elif role == "assistant":
                ant_content: List[dict] = []
                # text
                if isinstance(content, str) and content:
                    ant_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for b in content:
                        if b.get("type") == "text":
                            ant_content.append({"type": "text", "text": b["text"]})
                # tool calls on this assistant turn
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    ant_content.append({
                        "type": "tool_use",
                        "id":   tc["id"],
                        "name": fn.get("name", ""),
                        "input": ReqConv._parse_args(fn.get("arguments", {}))
                    })
                if ant_content:
                    messages_out.append({"role": "assistant", "content": ant_content})

            elif role == "tool":
                # Find or create a trailing user-tool_result block
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "")
                }
                # Merge consecutive tool results into one user message
                if (messages_out and messages_out[-1]["role"] == "user"
                        and isinstance(messages_out[-1]["content"], list)
                        and any(b.get("type") == "tool_result"
                                for b in messages_out[-1]["content"])):
                    messages_out[-1]["content"].append(tool_result)
                else:
                    messages_out.append({"role": "user", "content": [tool_result]})

        # System
        if system_parts:
            out["system"] = system_parts[0]["text"] if len(system_parts) == 1 \
                else system_parts

        out["messages"] = messages_out
        out["model"] = body.get("model", "")

        # max_tokens
        mt = body.get("max_tokens") or body.get("max_completion_tokens")
        out["max_tokens"] = mt or 4096

        # optional scalars
        for key in ("temperature", "top_p", "stop", "stream", "metadata"):
            if key in body:
                out[key] = body[key]
        if "stop" in out and isinstance(out["stop"], str):
            out["stop_sequences"] = [out.pop("stop")]
        elif "stop" in out and isinstance(out["stop"], list):
            out["stop_sequences"] = out.pop("stop")

        # tools
        oai_tools = body.get("tools") or []
        if oai_tools:
            ant_tools = []
            for t in oai_tools:
                if t.get("type") == "function":
                    fn = t["function"]
                    ant_tools.append({
                        "name": fn.get("name",""),
                        "description": fn.get("description",""),
                        "input_schema": fn.get("parameters",
                                               {"type":"object","properties":{}})
                    })
            out["tools"] = ant_tools

        # tool_choice
        tc = body.get("tool_choice")
        if tc:
            if tc == "auto":
                out["tool_choice"] = {"type": "auto"}
            elif tc == "required":
                out["tool_choice"] = {"type": "any"}
            elif tc == "none":
                out["tool_choice"] = {"type": "none"}
            elif isinstance(tc, dict) and tc.get("type") == "function":
                out["tool_choice"] = {"type": "tool",
                                      "name": tc["function"]["name"]}

        # strip OAI-only
        for f in ("frequency_penalty","presence_penalty","logit_bias","user","n",
                  "reasoning_effort","stream_options","response_format","logprobs",
                  "top_logprobs","seed","service_tier"):
            out.pop(f, None)

        return out

    # ── Anthropic → OpenAI ───────────────────────────────────────

    @staticmethod
    def anthropic_to_openai(body: dict) -> dict:
        out: dict = {}
        messages_out: List[dict] = []

        # system
        system = body.get("system")
        if system:
            text = system if isinstance(system, str) \
                else " ".join(b.get("text","") for b in system if isinstance(b,dict))
            messages_out.append({"role": "system", "content": text})

        for msg in body.get("messages", []):
            role    = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    messages_out.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    tool_results = [b for b in content if b.get("type") == "tool_result"]
                    normal = [b for b in content if b.get("type") != "tool_result"]

                    for tr in tool_results:
                        tc_content = tr.get("content","")
                        messages_out.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id",""),
                            "content": tc_content if isinstance(tc_content,str)
                                       else json.dumps(tc_content)
                        })
                    if normal:
                        parts = ReqConv._ant_content_to_oai_parts(normal)
                        if len(parts) == 1 and parts[0]["type"] == "text":
                            messages_out.append({"role":"user","content":parts[0]["text"]})
                        else:
                            messages_out.append({"role":"user","content":parts})

            elif role == "assistant":
                blocks = content if isinstance(content,list) else \
                    [{"type":"text","text":content}]
                text_parts = [b for b in blocks if b.get("type")=="text"]
                tool_uses  = [b for b in blocks if b.get("type")=="tool_use"]
                # thinking → convert to text comment
                think_parts= [b for b in blocks if b.get("type")=="thinking"]

                text_str = "".join(b["text"] for b in text_parts)
                for th in think_parts:
                    text_str += f"\n<thinking>{th.get('thinking','')}</thinking>"

                if tool_uses:
                    tool_calls = [{
                        "id":   tu["id"],
                        "type": "function",
                        "function": {
                            "name":      tu["name"],
                            "arguments": json.dumps(tu.get("input",{}))
                        }
                    } for tu in tool_uses]
                    messages_out.append({
                        "role": "assistant",
                        "content": text_str or None,
                        "tool_calls": tool_calls
                    })
                else:
                    messages_out.append({"role":"assistant","content":text_str})

        out["messages"] = messages_out
        out["model"]    = body.get("model","")

        mt = body.get("max_tokens")
        if mt: out["max_tokens"] = mt

        for key in ("temperature","top_p","stream"):
            if key in body: out[key] = body[key]
        if "stop_sequences" in body:
            out["stop"] = body["stop_sequences"]

        # tools
        ant_tools = body.get("tools") or []
        if ant_tools:
            out["tools"] = [{
                "type": "function",
                "function": {
                    "name":        t.get("name",""),
                    "description": t.get("description",""),
                    "parameters":  t.get("input_schema",{"type":"object","properties":{}})
                }
            } for t in ant_tools]

        # tool_choice
        tc = body.get("tool_choice")
        if tc:
            tt = tc.get("type","auto")
            if tt == "auto":   out["tool_choice"] = "auto"
            elif tt == "none": out["tool_choice"] = "none"
            elif tt == "any":  out["tool_choice"] = "required"
            elif tt == "tool": out["tool_choice"] = {
                "type":"function","function":{"name":tc.get("name","")}}

        # strip anthropic-only
        for f in ("thinking","betas","system","metadata","stop_sequences",
                  "top_k","anthropic_version"):
            out.pop(f,None)

        return out

    # ── OpenAI → Gemini ──────────────────────────────────────────

    @staticmethod
    def openai_to_gemini(body: dict) -> Tuple[dict, str]:
        """Returns (gemini_body, model_name)."""
        out: dict = {}
        model = body.get("model","gemini-2.0-flash")
        contents: List[dict] = []
        system_instruction = None

        for msg in body.get("messages",[]):
            role    = msg.get("role","")
            content = msg.get("content")

            if role in ("system","developer"):
                text = content if isinstance(content,str) \
                    else " ".join(b.get("text","") for b in (content or []) if isinstance(b,dict))
                system_instruction = {"parts":[{"text":text}]}

            elif role == "user":
                parts: List[dict] = []
                if isinstance(content, str):
                    parts = [{"text": content}]
                elif isinstance(content, list):
                    for b in content:
                        bt = b.get("type")
                        if bt == "text":
                            parts.append({"text": b["text"]})
                        elif bt == "image_url":
                            url = b["image_url"]["url"]
                            if url.startswith("data:"):
                                hdr, data = url.split(",",1)
                                mt = hdr.split(":")[1].split(";")[0]
                                parts.append({"inline_data":{"mime_type":mt,"data":data}})
                            else:
                                parts.append({"file_data":{"file_uri":url,
                                                            "mime_type":"image/jpeg"}})
                contents.append({"role":"user","parts":parts})

            elif role == "assistant":
                parts = []
                if content:
                    txt = content if isinstance(content,str) \
                        else "".join(b.get("text","") for b in content
                                     if isinstance(b,dict) and b.get("type")=="text")
                    if txt: parts.append({"text":txt})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function",{})
                    parts.append({"functionCall":{
                        "name": fn.get("name",""),
                        "args": ReqConv._parse_args(fn.get("arguments",{}))
                    }})
                if parts:
                    contents.append({"role":"model","parts":parts})

            elif role == "tool":
                # tool results → functionResponse in user turn
                if (contents and contents[-1]["role"] == "user"
                        and any("functionResponse" in p for p in contents[-1]["parts"])):
                    contents[-1]["parts"].append({"functionResponse":{
                        "name": msg.get("name", msg.get("tool_call_id","")),
                        "response":{"content": msg.get("content","")}
                    }})
                else:
                    contents.append({"role":"user","parts":[{"functionResponse":{
                        "name": msg.get("name", msg.get("tool_call_id","")),
                        "response":{"content": msg.get("content","")}
                    }}]})

        out["contents"] = contents
        if system_instruction:
            out["system_instruction"] = system_instruction

        # tools
        oai_tools = body.get("tools") or []
        if oai_tools:
            fn_decls = []
            for t in oai_tools:
                if t.get("type") == "function":
                    fn = t["function"]
                    fn_decls.append({
                        "name":        fn.get("name",""),
                        "description": fn.get("description",""),
                        "parameters":  fn.get("parameters",{})
                    })
            out["tools"] = [{"functionDeclarations": fn_decls}]

        # tool_choice → toolConfig
        tc = body.get("tool_choice")
        if tc:
            if tc == "none":
                out["tool_config"] = {"function_calling_config":{"mode":"NONE"}}
            elif tc == "required":
                out["tool_config"] = {"function_calling_config":{"mode":"ANY"}}
            elif isinstance(tc,dict) and tc.get("type")=="function":
                out["tool_config"] = {"function_calling_config":{
                    "mode":"ANY",
                    "allowed_function_names":[tc["function"]["name"]]
                }}

        # generation_config
        gc: dict = {}
        mt = body.get("max_tokens") or body.get("max_completion_tokens")
        if mt: gc["maxOutputTokens"] = mt
        if "temperature" in body: gc["temperature"] = body["temperature"]
        if "top_p" in body:       gc["topP"]        = body["top_p"]
        stop = body.get("stop")
        if stop: gc["stopSequences"] = [stop] if isinstance(stop,str) else stop
        if body.get("response_format",{}).get("type") == "json_object":
            gc["responseMimeType"] = "application/json"
        if gc: out["generation_config"] = gc

        return out, model

    # ── Gemini → OpenAI ──────────────────────────────────────────

    @staticmethod
    def gemini_to_openai(body: dict) -> dict:
        out: dict = {}
        messages: List[dict] = []

        si = body.get("system_instruction")
        if si:
            text = " ".join(p.get("text","") for p in si.get("parts",[]))
            messages.append({"role":"system","content":text})

        for c in body.get("contents",[]):
            role     = c.get("role","user")
            oai_role = "assistant" if role == "model" else "user"
            parts    = c.get("parts",[])

            text_parts = [p["text"] for p in parts if "text" in p]
            fn_calls   = [p["functionCall"]   for p in parts if "functionCall"   in p]
            fn_resp    = [p["functionResponse"] for p in parts if "functionResponse" in p]

            if fn_resp:
                for fr in fn_resp:
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": fr["name"],
                        "name":         fr["name"],
                        "content":      json.dumps(fr.get("response",{}))
                    })
            elif fn_calls:
                tool_calls = [{"id":fc["name"],"type":"function","function":{
                    "name":      fc["name"],
                    "arguments": json.dumps(fc.get("args",{}))
                }} for fc in fn_calls]
                messages.append({
                    "role":       oai_role,
                    "content":    " ".join(text_parts) or None,
                    "tool_calls": tool_calls
                })
            else:
                messages.append({"role":oai_role,"content":" ".join(text_parts)})

        out["messages"] = messages

        # tools back
        for tb in body.get("tools",[]):
            fn_decls = tb.get("functionDeclarations",[])
            if fn_decls:
                out["tools"] = [{"type":"function","function":{
                    "name":        fd["name"],
                    "description": fd.get("description",""),
                    "parameters":  fd.get("parameters",{})
                }} for fd in fn_decls]

        gc = body.get("generation_config",{})
        if "maxOutputTokens" in gc: out["max_tokens"]   = gc["maxOutputTokens"]
        if "temperature"     in gc: out["temperature"]  = gc["temperature"]
        if "topP"            in gc: out["top_p"]        = gc["topP"]
        if "stopSequences"   in gc: out["stop"]         = gc["stopSequences"]

        return out

    # ── bridged helpers ───────────────────────────────────────────

    @staticmethod
    def anthropic_to_gemini(body: dict) -> Tuple[dict, str]:
        return ReqConv.openai_to_gemini(ReqConv.anthropic_to_openai(body))

    @staticmethod
    def gemini_to_anthropic(body: dict) -> dict:
        return ReqConv.openai_to_anthropic(ReqConv.gemini_to_openai(body))


def convert_request(body: dict, from_fmt: str, to_fmt: str) -> Tuple[dict, str]:
    """
    Convert body from from_fmt to to_fmt.
    Returns (converted_body, model_str).
    model_str is only non-empty for gemini destination (needed in URL).
    """
    if from_fmt == to_fmt:
        model = body.get("model","")
        return body, model

    if from_fmt == "openai"     and to_fmt == "anthropic": return ReqConv.openai_to_anthropic(body), ""
    if from_fmt == "anthropic"  and to_fmt == "openai":    return ReqConv.anthropic_to_openai(body), ""
    if from_fmt == "openai"     and to_fmt == "gemini":
        b, m = ReqConv.openai_to_gemini(body);   return b, m
    if from_fmt == "gemini"     and to_fmt == "openai":    return ReqConv.gemini_to_openai(body), ""
    if from_fmt == "anthropic"  and to_fmt == "gemini":
        b, m = ReqConv.anthropic_to_gemini(body); return b, m
    if from_fmt == "gemini"     and to_fmt == "anthropic": return ReqConv.gemini_to_anthropic(body), ""

    return body, body.get("model","")


# ─────────────────────────────────────────────────────────────────
# RESPONSE CONVERTERS
# ─────────────────────────────────────────────────────────────────

class RespConv:
    """Convert a provider response body back to the client's expected format."""

    # ── to OpenAI ────────────────────────────────────────────────

    @staticmethod
    def anthropic_to_openai(r: dict) -> dict:
        blocks = r.get("content",[])
        texts  = [b["text"] for b in blocks if b.get("type")=="text"]
        tuses  = [b          for b in blocks if b.get("type")=="tool_use"]

        finish_map = {"end_turn":"stop","max_tokens":"length",
                      "tool_use":"tool_calls","stop_sequence":"stop"}
        finish = finish_map.get(r.get("stop_reason","end_turn"),"stop")

        if tuses:
            tool_calls = [{"id":tu["id"],"type":"function","function":{
                "name":      tu["name"],
                "arguments": json.dumps(tu.get("input",{}))
            }} for tu in tuses]
            msg = {"role":"assistant","content":"".join(texts) or None,
                   "tool_calls":tool_calls,"refusal":None}
        else:
            msg = {"role":"assistant","content":"".join(texts),"refusal":None}

        usage = r.get("usage",{})
        return {
            "id":      f"chatcmpl-{r.get('id','')}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   r.get("model",""),
            "choices": [{"index":0,"message":msg,"finish_reason":finish}],
            "usage":   {
                "prompt_tokens":     usage.get("input_tokens",0),
                "completion_tokens": usage.get("output_tokens",0),
                "total_tokens":      usage.get("input_tokens",0)+usage.get("output_tokens",0)
            }
        }

    @staticmethod
    def gemini_to_openai(r: dict, model: str) -> dict:
        choices = []
        for i, cand in enumerate(r.get("candidates",[])):
            parts   = cand.get("content",{}).get("parts",[])
            texts   = [p["text"] for p in parts if "text" in p and not p.get("thought")]
            fn_calls= [p["functionCall"] for p in parts if "functionCall" in p]
            if fn_calls:
                tool_calls = [{"id":fc["name"],"type":"function","function":{
                    "name":      fc["name"],
                    "arguments": json.dumps(fc.get("args",{}))
                }} for fc in fn_calls]
                msg    = {"role":"assistant","content":None,"tool_calls":tool_calls}
                finish = "tool_calls"
            else:
                msg = {"role":"assistant","content":"".join(texts)}
                fm  = {"STOP":"stop","MAX_TOKENS":"length","SAFETY":"content_filter"}
                finish = fm.get(cand.get("finishReason","STOP"),"stop")
            choices.append({"index":i,"message":msg,"finish_reason":finish})

        u = r.get("usageMetadata",{})
        return {
            "id":      f"chatcmpl-gemini-{int(time.time())}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   model,
            "choices": choices,
            "usage":   {
                "prompt_tokens":     u.get("promptTokenCount",0),
                "completion_tokens": u.get("candidatesTokenCount",0),
                "total_tokens":      u.get("totalTokenCount",0)
            }
        }

    # ── to Anthropic ─────────────────────────────────────────────

    @staticmethod
    def openai_to_anthropic(r: dict) -> dict:
        choices = r.get("choices",[])
        content: List[dict] = []
        stop_reason = "end_turn"
        if choices:
            msg  = choices[0].get("message",{})
            text = msg.get("content")
            if text: content.append({"type":"text","text":text})
            for tc in msg.get("tool_calls",[]) or []:
                fn = tc.get("function",{})
                content.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  fn.get("name",""),
                    "input": json.loads(fn["arguments"]) if isinstance(fn.get("arguments"),str)
                             else fn.get("arguments",{})
                })
            fm = {"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use","content_filter":"end_turn"}
            stop_reason = fm.get(choices[0].get("finish_reason","stop"),"end_turn")
        u = r.get("usage",{})
        return {
            "id":            f"msg_{r.get('id','')}",
            "type":          "message",
            "role":          "assistant",
            "content":       content,
            "model":         r.get("model",""),
            "stop_reason":   stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens":  u.get("prompt_tokens",0),
                "output_tokens": u.get("completion_tokens",0)
            }
        }

    @staticmethod
    def gemini_to_anthropic(r: dict, model: str) -> dict:
        return RespConv.openai_to_anthropic(RespConv.gemini_to_openai(r, model))

    # ── to Gemini ────────────────────────────────────────────────

    @staticmethod
    def openai_to_gemini(r: dict) -> dict:
        candidates = []
        for i, ch in enumerate(r.get("choices",[])):
            msg   = ch.get("message",{})
            parts = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg.get("tool_calls",[]) or []:
                fn = tc.get("function",{})
                parts.append({"functionCall":{
                    "name": fn.get("name",""),
                    "args": json.loads(fn["arguments"]) if isinstance(fn.get("arguments"),str)
                            else fn.get("arguments",{})
                }})
            fm = {"stop":"STOP","length":"MAX_TOKENS","tool_calls":"STOP","content_filter":"SAFETY"}
            candidates.append({
                "content":      {"parts":parts,"role":"model"},
                "finishReason": fm.get(ch.get("finish_reason","stop"),"STOP"),
                "index":        i
            })
        u = r.get("usage",{})
        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount":     u.get("prompt_tokens",0),
                "candidatesTokenCount": u.get("completion_tokens",0),
                "totalTokenCount":      u.get("total_tokens",0)
            }
        }

    @staticmethod
    def anthropic_to_gemini(r: dict) -> dict:
        return RespConv.openai_to_gemini(RespConv.anthropic_to_openai(r))

    # ── dispatcher ───────────────────────────────────────────────

    @staticmethod
    def convert(resp: dict, provider_fmt: str, client_fmt: str, model: str = "") -> dict:
        if provider_fmt == client_fmt:
            return resp
        # normalise to openai first
        if provider_fmt == "anthropic":
            oai = RespConv.anthropic_to_openai(resp)
        elif provider_fmt == "gemini":
            oai = RespConv.gemini_to_openai(resp, model)
        else:
            oai = resp

        if client_fmt == "openai":    return oai
        if client_fmt == "anthropic": return RespConv.openai_to_anthropic(oai)
        if client_fmt == "gemini":    return RespConv.openai_to_gemini(oai)
        return oai


# ─────────────────────────────────────────────────────────────────
# STREAMING
# ─────────────────────────────────────────────────────────────────

def _parse_sse(raw: str) -> List[Tuple[str, Any]]:
    """Parse raw SSE text → [(event_name, parsed_data_dict|None)]."""
    events: List[Tuple[str,Any]] = []
    current_event = "message"
    for line in raw.splitlines():
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            ds = line[5:].strip()
            if ds == "[DONE]":
                events.append(("done", None))
                current_event = "message"
            else:
                try:
                    events.append((current_event, json.loads(ds)))
                except Exception:
                    pass
                current_event = "message"
        elif line == "":
            current_event = "message"
    return events


def _reconstruct_from_events(events: List[Tuple[str,Any]],
                              provider_fmt: str) -> Tuple[str, List[dict], str]:
    """
    Walk SSE events and reconstruct:
      text        - accumulated text
      tool_calls  - list of tool_call objects (OAI format)
      stop_reason - 'end_turn' | 'max_tokens' | 'tool_use'
    """
    text = ""
    tool_calls: Dict[int, dict] = {}  # index → {id, name, args_str}
    stop_reason = "end_turn"

    for ev_name, data in events:
        if data is None:
            continue

        if provider_fmt == "openai":
            for ch in data.get("choices",[]):
                delta  = ch.get("delta",{})
                finish = ch.get("finish_reason")
                if delta.get("content"):
                    text += delta["content"]
                # stream tool_calls accumulation
                for tc_delta in delta.get("tool_calls",[]) or []:
                    idx = tc_delta.get("index",0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id":"","name":"","args":""}
                    if tc_delta.get("id"):
                        tool_calls[idx]["id"] = tc_delta["id"]
                    fn = tc_delta.get("function",{})
                    if fn.get("name"):
                        tool_calls[idx]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_calls[idx]["args"] += fn["arguments"]
                if finish:
                    fm = {"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use"}
                    stop_reason = fm.get(finish,"end_turn")

        elif provider_fmt == "anthropic":
            if ev_name == "content_block_delta":
                d = data.get("delta",{})
                if d.get("type") == "text_delta":
                    text += d.get("text","")
                elif d.get("type") == "input_json_delta":
                    idx = data.get("index",0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id":"","name":"","args":""}
                    tool_calls[idx]["args"] += d.get("partial_json","")
            elif ev_name == "content_block_start":
                cb = data.get("content_block",{})
                if cb.get("type") == "tool_use":
                    idx = data.get("index",0)
                    tool_calls[idx] = {"id":cb.get("id",""),
                                       "name":cb.get("name",""),"args":""}
            elif ev_name == "message_delta":
                stop_reason = data.get("delta",{}).get("stop_reason","end_turn") or "end_turn"

        elif provider_fmt == "gemini":
            for cand in data.get("candidates",[]):
                for part in cand.get("content",{}).get("parts",[]):
                    if "text" in part and not part.get("thought"):
                        text += part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        idx = len(tool_calls)
                        tool_calls[idx] = {
                            "id":   fc["name"],
                            "name": fc["name"],
                            "args": json.dumps(fc.get("args",{}))
                        }
                fr = cand.get("finishReason","")
                if fr == "STOP": stop_reason = "end_turn"

    # Normalise tool_calls list
    tc_list = []
    for idx in sorted(tool_calls.keys()):
        tc = tool_calls[idx]
        tc_list.append({
            "id":   tc["id"] or f"tc_{idx}",
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["args"]}
        })

    return text, tc_list, stop_reason


def _emit_openai_stream(text: str, tool_calls: List[dict],
                         stop_reason: str, model: str) -> str:
    mid = f"chatcmpl-proxy-{int(time.time())}"
    out = ""
    # role chunk
    out += f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"
    if text:
        out += f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'content':text},'finish_reason':None}]})}\n\n"
    if tool_calls:
        # emit tool_call deltas
        for i, tc in enumerate(tool_calls):
            out += f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':i,'id':tc['id'],'type':'function','function':{'name':tc['function']['name'],'arguments':''}}]},'finish_reason':None}]})}\n\n"
            out += f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':i,'function':{'arguments':tc['function']['arguments']}}]},'finish_reason':None}]})}\n\n"
    fm = {"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls"}
    out += f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':fm.get(stop_reason,'stop')}]})}\n\n"
    out += "data: [DONE]\n\n"
    return out


def _emit_anthropic_stream(text: str, tool_calls: List[dict],
                            stop_reason: str, model: str) -> str:
    mid = f"msg_proxy_{int(time.time())}"
    out = ""
    out += f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':mid,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"

    block_idx = 0
    if text:
        out += f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':block_idx,'content_block':{'type':'text','text':''}})}\n\n"
        out += f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':block_idx,'delta':{'type':'text_delta','text':text}})}\n\n"
        out += f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':block_idx})}\n\n"
        block_idx += 1

    for tc in tool_calls:
        fn = tc["function"]
        out += f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':block_idx,'content_block':{'type':'tool_use','id':tc['id'],'name':fn['name'],'input':{}}})}\n\n"
        out += f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':block_idx,'delta':{'type':'input_json_delta','partial_json':fn['arguments']}})}\n\n"
        out += f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':block_idx})}\n\n"
        block_idx += 1

    out += f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    out += f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
    return out


def _emit_gemini_stream(text: str, tool_calls: List[dict],
                         stop_reason: str, model: str) -> str:
    parts: List[dict] = []
    if text:
        parts.append({"text": text})
    for tc in tool_calls:
        fn = tc["function"]
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"],str) else fn["arguments"]
        parts.append({"functionCall":{"name":fn["name"],"args":args}})
    return f"data: {json.dumps({'candidates':[{'content':{'parts':parts,'role':'model'},'finishReason':'STOP','index':0}]})}\n\n"


def convert_stream(raw_sse: str, provider_fmt: str, client_fmt: str, model: str) -> str:
    """Buffer-and-convert: parse upstream SSE, reconstruct, re-emit in client format."""
    events = _parse_sse(raw_sse)
    text, tool_calls, stop_reason = _reconstruct_from_events(events, provider_fmt)

    if client_fmt == "openai":
        return _emit_openai_stream(text, tool_calls, stop_reason, model)
    elif client_fmt == "anthropic":
        return _emit_anthropic_stream(text, tool_calls, stop_reason, model)
    elif client_fmt == "gemini":
        return _emit_gemini_stream(text, tool_calls, stop_reason, model)
    return raw_sse


# ─────────────────────────────────────────────────────────────────
# URL BUILDER
# ─────────────────────────────────────────────────────────────────

def build_url(provider: dict, client_path: str, provider_fmt: str,
              model: str, is_stream: bool) -> str:
    base = provider["base_url"].rstrip("/")

    if provider_fmt == "gemini":
        action = "streamGenerateContent" if is_stream else "generateContent"
        m = model or "gemini-2.0-flash"
        url = f"{base}/v1beta/models/{m}:{action}"
        if is_stream:
            url += "?alt=sse"
        return url

    if provider_fmt == "anthropic":
        return f"{base}/v1/messages"

    # openai-compatible — respect the client's path when it makes sense
    pl = client_path.split("?")[0].lower()
    if "/v1/chat/completions" in pl: return f"{base}/v1/chat/completions"
    if "/api/chat"            in pl: return f"{base}/api/chat"
    if "/v1/completions"      in pl: return f"{base}/v1/completions"
    if "/v1/responses"        in pl: return f"{base}/v1/chat/completions"  # translate new→old
    return f"{base}/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────
# PROXY HANDLER
# ─────────────────────────────────────────────────────────────────

_cfg:  Optional[Config]            = None
_args: Optional[argparse.Namespace]= None

class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, *_): pass  # silence default

    # ── routing ──────────────────────────────────────────────────

    def do_GET(self):    self._passthrough()
    def do_DELETE(self): self._passthrough()
    def do_POST(self):   self._llm_request()
    def do_PUT(self):    self._passthrough()

    # ── GET/DELETE passthrough ───────────────────────────────────

    def _passthrough(self):
        provider = self._get_provider()
        if not provider: return
        pool = get_pool(provider)
        key  = pool.next_key()
        if not key:
            self._error(429, "All keys on cooldown"); return

        target = f"{provider['base_url'].rstrip('/')}{self.path}"
        hdrs   = self._build_headers(provider, key, {})
        req    = urllib.request.Request(target, headers=hdrs, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=_cfg.timeout) as r:
                self.send_response(r.getcode())
                for h,v in r.getheaders():
                    if h.lower() not in ("connection","transfer-encoding"):
                        self.send_header(h,v)
                self.end_headers()
                self.wfile.write(r.read())
        except Exception as e:
            self._error(502, str(e))

    # ── main LLM handler ─────────────────────────────────────────

    def _llm_request(self):
        ts = datetime.now().strftime("%H:%M:%S")

        # 1. Read body
        try:
            length   = int(self.headers.get("Content-Length",0))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            body     = json.loads(raw_body) if raw_body.strip() else {}
        except Exception as e:
            self._error(400, f"Bad JSON: {e}"); return

        # 2. Resolve provider
        provider = self._get_provider()
        if not provider: return

        provider_fmt = provider["native_format"]

        # 3. Detect client format
        client_fmt = _args.from_format if _args.from_format != "auto" \
                     else detect_client_format(self.path, dict(self.headers), body)

        # 4. Decide response format to send back
        reply_fmt = _args.to_format or client_fmt

        # 5. Model
        orig_model     = body.get("model","")
        resolved_model = resolve_model(orig_model, provider)
        is_stream      = bool(body.get("stream", False))

        if orig_model != resolved_model:
            body = copy.deepcopy(body)
            body["model"] = resolved_model

        log(f"\n[{ts}] {BOLD}POST{RESET} {self.path}")
        log(f"  {BLUE}Client fmt  :{RESET} {client_fmt.upper()}"
            + (f" (reply as {reply_fmt.upper()})" if reply_fmt != client_fmt else ""))
        log(f"  {MAGENTA}Provider    :{RESET} {provider['name']} [{provider_fmt.upper()}] → {provider['base_url']}")
        log(f"  {CYAN}Model       :{RESET} {orig_model or '(none)'}"
            + (f" → {GREEN}{resolved_model}{RESET}" if orig_model != resolved_model else ""))
        log(f"  {GREY}Stream      : {is_stream}{RESET}")

        # 6. Convert request
        try:
            converted, gemini_model = convert_request(body, client_fmt, provider_fmt)
            gemini_model = gemini_model or resolved_model
        except Exception as e:
            self._error(500, f"Request conversion failed: {e}\n{traceback.format_exc()}")
            return

        target_url = build_url(provider, self.path, provider_fmt, gemini_model, is_stream)
        vlog(f"  Target URL  : {target_url}")

        # 7. Fire with retry / key-rotation
        self._fire(
            provider=provider,
            converted=converted,
            target_url=target_url,
            client_fmt=client_fmt,
            reply_fmt=reply_fmt,
            provider_fmt=provider_fmt,
            is_stream=is_stream,
            model=gemini_model or resolved_model
        )

    # ── fire + retry ─────────────────────────────────────────────

    def _fire(self, provider, converted, target_url, client_fmt, reply_fmt,
              provider_fmt, is_stream, model):
        pool    = get_pool(provider)
        retries = _cfg.max_retries

        for attempt in range(retries):
            key = pool.next_key()
            if key is None:
                self._error(429, "All API keys are on cooldown."); return

            log(f"  {GREEN}Key #{attempt+1}{RESET} …{key[-10:]}")
            headers  = self._build_headers(provider, key, converted)
            payload  = json.dumps(converted).encode("utf-8")
            req      = urllib.request.Request(
                url=target_url, data=payload, headers=headers, method="POST")

            try:
                resp   = urllib.request.urlopen(req, timeout=_cfg.timeout)
                status = resp.getcode()
                log(f"  {GREEN}✓ {status}{RESET}")
                self._forward(resp, client_fmt, reply_fmt, provider_fmt, is_stream, model)
                return

            except urllib.error.HTTPError as e:
                log(f"  {YELLOW}⚠ HTTP {e.code} (attempt {attempt+1}/{retries}){RESET}")
                if e.code == 429:
                    try:
                        ra = int(e.headers.get("Retry-After", 60))
                    except Exception:
                        ra = 60
                    pool.rate_limit(key, ra)
                elif e.code in (401, 403):
                    pool.invalidate(key)
                elif e.code not in _cfg.retry_codes:
                    # non-retryable — forward as-is
                    self.send_response(e.code)
                    for h,v in e.headers.items():
                        if h.lower() not in ("connection","transfer-encoding"):
                            self.send_header(h,v)
                    self.end_headers()
                    body = e.read()
                    # try to convert error too
                    try:
                        err_body = json.loads(body)
                        if reply_fmt == "anthropic" and provider_fmt == "openai":
                            pass  # keep as-is (error shape differs)
                    except Exception:
                        pass
                    self.wfile.write(body)
                    return

            except Exception as e:
                log(f"  {RED}✗ {type(e).__name__}: {e}{RESET}")
                if attempt == retries - 1:
                    self._error(502, f"All retries failed: {e}")
                    return

        self._error(502, f"All {retries} attempts failed.")

    # ── forward response ─────────────────────────────────────────

    def _forward(self, resp, client_fmt, reply_fmt, provider_fmt, is_stream, model):
        if is_stream:
            self._forward_stream(resp, client_fmt, reply_fmt, provider_fmt, model)
        else:
            self._forward_json(resp, client_fmt, reply_fmt, provider_fmt, model)

    def _forward_json(self, resp, client_fmt, reply_fmt, provider_fmt, model):
        raw = resp.read()
        try:
            body = json.loads(raw)
        except Exception:
            self.send_response(200)
            for h,v in resp.getheaders():
                if h.lower() not in ("connection","transfer-encoding","content-length"):
                    self.send_header(h,v)
            self.end_headers()
            self.wfile.write(raw)
            return

        # convert response from provider_fmt → reply_fmt
        if provider_fmt != reply_fmt:
            try:
                body = RespConv.convert(body, provider_fmt, reply_fmt, model)
            except Exception as e:
                log(f"  {YELLOW}⚠ Response conversion error: {e}{RESET}")

        out = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _forward_stream(self, resp, client_fmt, reply_fmt, provider_fmt, model):
        same = (provider_fmt == reply_fmt)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")

        if same:
            # Pure passthrough — stream directly
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk: break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception:
                pass
            return

        # Buffer upstream → convert → emit
        buf = b""
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk: break
                buf += chunk
        except Exception as e:
            log(f"  {YELLOW}⚠ Stream read: {e}{RESET}")

        raw_text = buf.decode("utf-8", errors="replace")
        converted_sse = convert_stream(raw_text, provider_fmt, reply_fmt, model)
        out = converted_sse.encode("utf-8")

        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
        self.wfile.flush()

    # ── helpers ──────────────────────────────────────────────────

    def _get_provider(self) -> Optional[dict]:
        name = _args.provider or _cfg.default_provider
        p = _cfg.get(name)
        if not p:
            avail = list(_cfg.providers.keys())
            self._error(400, f"Provider '{name}' not found or not enabled. "
                            f"Available: {avail}")
            return None
        return p

    def _build_headers(self, provider: dict, key: str, body: dict) -> dict:
        fmt = provider["native_format"]
        hdrs: dict = {}

        # Content-type only if we have a body to send
        if body:
            hdrs["Content-Type"] = "application/json"

        # Auth
        if fmt == "anthropic":
            hdrs["x-api-key"]          = key
            hdrs["anthropic-version"] = "2023-06-01"
        elif fmt == "gemini":
            hdrs["x-goog-api-key"] = key
        else:
            hdrs["Authorization"] = f"Bearer {key}"

        # Host
        hdrs["Host"] = urllib.parse.urlparse(provider["base_url"]).netloc

        # Extra headers from config (can override anything above)
        hdrs.update(provider.get("extra_headers", {}))

        # Remove incoming auth / host headers
        for k in list(self.headers.keys()):
            kl = k.lower()
            if kl in ("host","authorization","x-api-key","x-goog-api-key",
                       "content-length","content-type","connection","keep-alive"):
                continue
            hdrs[k] = self.headers[k]

        return hdrs

    def _error(self, code: int, msg: str):
        log(f"  {RED}✗ {code} — {msg}{RESET}")
        body = json.dumps({"error":{"message":msg,"type":"proxy_error","code":code}}).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    global _cfg, _args, _verbose

    _args    = parse_args()
    _verbose = _args.verbose
    _cfg     = Config(_args.config)

    host = _args.host or _cfg.server.get("host","localhost")
    port = _args.port or _cfg.server.get("port", 8000)

    # validate provider
    pname = _args.provider or _cfg.default_provider
    if pname not in _cfg.providers:
        log(f"\n{RED}✗ Provider '{pname}' not enabled.{RESET}")
        log(f"  Enabled providers: {list(_cfg.providers.keys())}")
        sys.exit(1)

    provider    = _cfg.providers[pname]
    native_fmt  = provider["native_format"]
    reply_label = f"→ {_args.to_format.upper()}" if _args.to_format else "(mirrors client)"

    print(f"""
{BOLD}{'═'*62}{RESET}
{BOLD}  🔀  Universal LLM Proxy{RESET}
{'═'*62}
  Listen     : http://{host}:{port}
  Provider   : {CYAN}{pname}{RESET}  [{native_fmt.upper()} format]
  Target URL : {provider['base_url']}
  From       : {_args.from_format.upper()}  (auto-detect if 'auto')
  To (reply) : {reply_label}
  API Keys   : {len(provider.get('api_keys',[]))} loaded
  Max Retry  : {_cfg.max_retries}
{'═'*62}
  Client example:
    base_url = "http://{host}:{port}"
    api_key  = "any-string"   ← proxy ignores client key
{'═'*62}
""")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, port), ProxyHandler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}  ⏹  Proxy stopped.{RESET}\n")


if __name__ == "__main__":
    main()

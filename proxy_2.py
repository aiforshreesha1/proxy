#!/usr/bin/env python3
"""
Universal LLM Proxy  ─  v2  (OAuth + API-Key edition)
======================================================
Auto-detects client request format (OpenAI / Anthropic / Gemini).
Converts to the target provider's native format. Converts response back.

THREE credential modes (can be mixed inside one provider):
  api_key   – classic API key in Authorization: Bearer / x-api-key / x-goog-api-key
  oauth     – JSON token files on disk with access_token + refresh_token + auto-refresh
  bearer    – raw bearer tokens with NO auto-refresh (useful for iFlow/cookie tokens)

How the free OAuth providers work (what was researched):
  Claude Code   → PKCE OAuth on port 54545 → sk-ant-oat01-… token
  OpenAI Codex  → PKCE OAuth on port 1455  → OpenAI session token
  Gemini CLI    → Google OAuth on port 8085 → Google access_token
  Qwen Code     → Device-flow OAuth         → Qwen access_token
  iFlow         → Browser cookie → API token exchange

Token files follow this shape (any subset is fine):
  {
    "access_token":  "sk-ant-oat01-...",
    "refresh_token": "sk-ant-ort01-...",
    "expires_at":    1735000000.0,        ← Unix timestamp (float or int)
    "token_type":    "Bearer",
    "provider":      "claude",            ← optional hint
    "email":         "user@example.com"   ← optional hint
  }

Usage:
  python proxy.py                              # use default provider from config
  python proxy.py --provider claude_oauth      # use the Claude-OAuth provider
  python proxy.py --provider gemini            # use Gemini
  python proxy.py --provider ollama --to anthropic
  python proxy.py --provider openrouter --from anthropic
  python proxy.py --port 9000 --config my.json
  python proxy.py --verbose
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

# ─── ANSI colours ────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD    = "\033[1m";  CYAN  = "\033[96m"
GREEN  = "\033[92m"; YELLOW  = "\033[93m"; RED   = "\033[91m"
GREY   = "\033[90m"; BLUE    = "\033[94m"; MAGENTA = "\033[95m"

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
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal LLM Format-Converting Proxy (API-key + OAuth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python proxy.py --provider groq
  python proxy.py --provider claude_oauth       # OAuth token files
  python proxy.py --provider ollama --to anthropic
  python proxy.py --provider openrouter --from anthropic
  python proxy.py --provider gemini --port 9000
""")
    p.add_argument("--provider", default=None,
                   help="Backend provider name (matches 'name' in config). Default: config defaults.provider")
    p.add_argument("--from", dest="from_format", default="auto",
                   choices=["auto","openai","anthropic","gemini"],
                   help="Client request format hint. Default: auto-detect")
    p.add_argument("--to", dest="to_format", default=None,
                   choices=["openai","anthropic","gemini"],
                   help="Force response format sent back to client. Default: mirrors client format")
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
# OAUTH TOKEN FILE POOL
# Reads JSON token files from disk, auto-refreshes expired tokens.
# ─────────────────────────────────────────────────────────────────

# Known token refresh endpoints per provider name
_REFRESH_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "claude": {
        "token_endpoint": "https://claude.ai/api/oauth/token",
        "client_id":      "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
    },
    "codex": {
        "token_endpoint": "https://auth.openai.com/oauth/token",
        "client_id":      "app_EMOhFvvMY2qAMolFAJO2bEip",  # OpenAI CLI client
    },
    "gemini": {
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "client_id":      "",  # Google OAuth – use the client_id stored in the token file
    },
    "qwen": {
        "token_endpoint": "https://account.aliyun.com/oauth2/token",
        "client_id":      "",
    },
    "iflow": {
        # iFlow uses cookie→token exchange – no standard refresh; bearer tokens only
        "token_endpoint": "",
        "client_id":      "",
    },
    "antigravity": {
        "token_endpoint": "https://sso.hailuo.ai/oauth2/token",
        "client_id":      "",
    },
}

class TokenFile:
    """Wraps a single OAuth JSON token file with refresh support."""

    def __init__(self, path: str, provider_name: str = ""):
        self.path = os.path.expanduser(path)
        self.provider_name = provider_name
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()

    # ── Known nested-key wrappers in various CLI tools ──────────────
    _NESTED_KEYS = [
        "claudeAiOauthToken",   # ~/.claude.json  (Claude Code)
        "credential",           # Google oauth_creds.json variant
        "token",                # some Gemini variants
        "auth",                 # generic wrapper
    ]

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                # Flatten nested token objects (e.g. ~/.claude.json uses claudeAiOauthToken)
                self._data = self._flatten(raw)
            except Exception as e:
                log(f"  {YELLOW}⚠ Token file read error ({self.path}): {e}{RESET}")
                self._data = {}

    def _flatten(self, raw: dict) -> dict:
        """
        If the JSON is a single-key wrapper around the real token, unwrap it.
        Examples:
          { "claudeAiOauthToken": { "access_token": "sk-ant-..." } }
          → { "access_token": "sk-ant-...", "_source_key": "claudeAiOauthToken" }
        """
        # Already has access_token at top level → done
        if "access_token" in raw or "token" in raw:
            return raw
        # Check known wrapper keys
        for k in self._NESTED_KEYS:
            if k in raw and isinstance(raw[k], dict):
                inner = dict(raw[k])
                inner["_source_key"] = k   # remember for save-back
                inner["_outer"] = raw      # keep outer for full re-serialisation
                return inner
        # Fallback: check any key whose value is a dict with access_token
        for k, v in raw.items():
            if isinstance(v, dict) and ("access_token" in v or "token" in v):
                inner = dict(v)
                inner["_source_key"] = k
                inner["_outer"] = raw
                return inner
        return raw

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # If we unwrapped a nested key, re-wrap before saving
            source_key = self._data.get("_source_key")
            if source_key:
                outer = dict(self._data.get("_outer", {}))
                inner = {k: v for k, v in self._data.items()
                         if k not in ("_source_key","_outer")}
                outer[source_key] = inner
                out = outer
            else:
                out = {k: v for k, v in self._data.items()
                       if k not in ("_source_key","_outer")}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            log(f"  {YELLOW}⚠ Token file write error ({self.path}): {e}{RESET}")

    @property
    def access_token(self) -> Optional[str]:
        return self._data.get("access_token") or self._data.get("token")

    @property
    def refresh_token(self) -> Optional[str]:
        return self._data.get("refresh_token")

    @property
    def expires_at(self) -> float:
        raw = self._data.get("expires_at", 0)
        return float(raw) if raw else 0.0

    @property
    def is_expired(self) -> bool:
        """True if token expires in less than 5 minutes."""
        ea = self.expires_at
        if ea == 0:
            return False  # no expiry info → assume valid
        return time.time() >= (ea - 300)

    @property
    def is_valid(self) -> bool:
        return bool(self.access_token)

    def _detect_provider(self) -> str:
        """Guess provider from stored hint or file path."""
        p = self._data.get("provider", "").lower()
        if p: return p
        base = os.path.basename(self.path).lower()
        for name in _REFRESH_ENDPOINTS:
            if name in base:
                return name
        return self.provider_name.lower()

    def refresh(self) -> bool:
        """
        Attempt token refresh using refresh_token.
        Returns True if successful.
        """
        rt = self.refresh_token
        if not rt:
            log(f"  {YELLOW}⚠ No refresh_token in {os.path.basename(self.path)}{RESET}")
            return False

        provider = self._detect_provider()
        meta     = _REFRESH_ENDPOINTS.get(provider, {})

        # Allow per-file override of endpoint / client_id
        token_endpoint = (self._data.get("_meta", {}).get("token_endpoint")
                          or meta.get("token_endpoint", ""))
        client_id      = (self._data.get("_meta", {}).get("client_id")
                          or self._data.get("client_id", "")
                          or meta.get("client_id", ""))

        if not token_endpoint:
            log(f"  {YELLOW}⚠ No token_endpoint for provider '{provider}' – cannot refresh{RESET}")
            return False

        payload = urllib.parse.urlencode({
            "grant_type":    "refresh_token",
            "refresh_token": rt,
            "client_id":     client_id,
        }).encode("utf-8")

        req = urllib.request.Request(
            token_endpoint,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except Exception as e:
            log(f"  {RED}✗ Token refresh failed ({os.path.basename(self.path)}): {e}{RESET}")
            return False

        if "access_token" not in body:
            log(f"  {RED}✗ Refresh response missing access_token: {body}{RESET}")
            return False

        # Merge new tokens into existing data
        self._data["access_token"] = body["access_token"]
        if "refresh_token" in body:
            self._data["refresh_token"] = body["refresh_token"]
        expires_in = body.get("expires_in", 28800)  # default 8h
        self._data["expires_at"] = time.time() + expires_in
        self._data["token_type"] = body.get("token_type", "Bearer")
        self._save()

        email = self._data.get("email", os.path.basename(self.path))
        log(f"  {GREEN}✓ Token refreshed for {email}{RESET}")
        return True

    def get_token(self) -> Optional[str]:
        """
        Return a valid access token, refreshing first if needed.
        Thread-safe.
        """
        with self._lock:
            if self.is_expired:
                log(f"  {YELLOW}⏳ Token expired ({os.path.basename(self.path)}), refreshing...{RESET}")
                if not self.refresh():
                    return None
            return self.access_token


class TokenFilePool:
    """
    Round-robin pool of TokenFile objects with per-file cooldown.
    Supports multiple OAuth accounts.
    """

    def __init__(self, paths: List[str], provider_name: str = ""):
        self._files: List[TokenFile] = [TokenFile(p, provider_name) for p in paths]
        self._idx   = 0
        self._lock  = threading.Lock()
        self._cooldown: Dict[str, float] = {}  # path → cool_until

    def next_credential(self) -> Optional[Tuple[str, str]]:
        """Returns (token, source_path) or None if all on cooldown."""
        with self._lock:
            now = time.time()
            for _ in range(len(self._files)):
                tf = self._files[self._idx]
                self._idx = (self._idx + 1) % len(self._files)
                if now < self._cooldown.get(tf.path, 0):
                    continue
                tok = tf.get_token()
                if tok:
                    return tok, tf.path
        return None

    def rate_limit(self, path: str, seconds: int = 60):
        with self._lock:
            self._cooldown[path] = time.time() + seconds
        log(f"  ⏳ OAuth token {os.path.basename(path)} rate-limited {seconds}s", YELLOW)

    def invalidate(self, path: str, seconds: int = 3600):
        with self._lock:
            self._cooldown[path] = time.time() + seconds
        log(f"  ❌ OAuth token {os.path.basename(path)} invalidated for {seconds//60}m", RED)

    def __len__(self):
        return len(self._files)

# ─────────────────────────────────────────────────────────────────
# API KEY POOL  (classic key rotation)
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
        return None

    def rate_limit(self, key: str, retry_after: int = 60):
        with self._lock:
            self._cooldown[key] = time.time() + retry_after
        log(f"  ⏳ Key …{key[-8:]} rate-limited {retry_after}s", YELLOW)

    def invalidate(self, key: str, seconds: int = 3600):
        with self._lock:
            self._cooldown[key] = time.time() + seconds
        log(f"  ❌ Key …{key[-8:]} invalid {seconds//60}m", RED)

# ─────────────────────────────────────────────────────────────────
# UNIFIED CREDENTIAL  (api_key  |  oauth_file  |  bearer)
# ─────────────────────────────────────────────────────────────────

class Credential:
    """
    Represents one usable credential: a key string + its source type + its
    pool (so we can call pool.rate_limit / pool.invalidate on the right object).
    """
    __slots__ = ("token", "cred_type", "source", "_pool_ref")

    def __init__(self, token: str, cred_type: str, source: str, pool_ref):
        self.token     = token       # the raw bearer / api-key string
        self.cred_type = cred_type   # "api_key" | "oauth" | "bearer"
        self.source    = source      # key itself OR file path
        self._pool_ref = pool_ref

    def rate_limit(self, seconds: int = 60):
        self._pool_ref.rate_limit(self.source, seconds)

    def invalidate(self, seconds: int = 3600):
        self._pool_ref.invalidate(self.source, seconds)


_cred_pools: Dict[str, Any] = {}   # provider_name → CredentialManager
_cp_lock = threading.Lock()

class CredentialManager:
    """
    Wraps KeyPool + TokenFilePool + bearer pool for one provider.
    next_credential() returns a Credential in round-robin across all types.
    """

    def __init__(self, provider: dict):
        self._lock    = threading.Lock()
        self._sources = []  # list of (pool_object, type_label)

        keys = provider.get("api_keys") or []
        if keys:
            kp = KeyPool(keys)
            self._sources.append((kp, "api_key"))

        token_files = provider.get("oauth_token_files") or []
        if token_files:
            tp = TokenFilePool(token_files, provider.get("name", ""))
            self._sources.append((tp, "oauth"))

        bearer_tokens = provider.get("oauth_bearer_tokens") or []
        if bearer_tokens:
            bp = KeyPool(bearer_tokens)  # KeyPool works fine for raw bearers too
            self._sources.append((bp, "bearer"))

        # index cycling across source types
        self._src_idx = 0

    def next_credential(self) -> Optional[Credential]:
        with self._lock:
            n = len(self._sources)
            for _ in range(n):
                pool, ctype = self._sources[self._src_idx]
                self._src_idx = (self._src_idx + 1) % n

                if ctype == "oauth":
                    result = pool.next_credential()
                    if result:
                        tok, path = result
                        return Credential(tok, ctype, path, pool)
                else:
                    key = pool.next_key()
                    if key:
                        return Credential(key, ctype, key, pool)
        return None

    @property
    def total(self) -> int:
        total = 0
        for pool, _ in self._sources:
            if hasattr(pool, "_files"):
                total += len(pool)
            else:
                total += len(pool._keys)
        return total


def get_cred_mgr(provider: dict) -> CredentialManager:
    name = provider["name"]
    with _cp_lock:
        if name not in _cred_pools:
            _cred_pools[name] = CredentialManager(provider)
    return _cred_pools[name]

# ─────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_client_format(path: str, headers: Dict[str, str], body: dict) -> str:
    pl = path.lower()
    if "/v1/messages"          in pl: return "anthropic"
    if "/v1beta/"              in pl or "generatecontent" in pl: return "gemini"
    if "/v1/chat/completions"  in pl or "/v1/completions" in pl: return "openai"
    if "/v1/responses"         in pl: return "openai"

    lh = {k.lower() for k in headers}
    if "anthropic-version" in lh or "x-api-key" in lh: return "anthropic"
    if "x-goog-api-key"    in lh:                       return "gemini"

    if "contents" in body or "system_instruction" in body: return "gemini"
    if "system" in body and "messages" in body:
        msgs = body.get("messages", [])
        if msgs and msgs[0].get("role") not in ("system","developer"):
            return "anthropic"
    if "thinking" in body or "anthropic_version" in body: return "anthropic"

    return "openai"

# ─────────────────────────────────────────────────────────────────
# MODEL RESOLUTION
# ─────────────────────────────────────────────────────────────────

def resolve_model(requested: str, provider: dict) -> str:
    model_map = provider.get("model_map") or {}
    if requested in model_map:
        mapped = model_map[requested]
        if mapped != requested:
            log(f"  🔀 model_map: {CYAN}{requested}{RESET} → {GREEN}{mapped}{RESET}")
        return mapped

    supported = provider.get("supported_models") or []
    if not supported or requested in supported:
        return requested

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
    @staticmethod
    def _oai_content_to_anthropic_parts(content) -> List[dict]:
        if isinstance(content, str):
            return [{"type":"text","text":content}]
        parts = []
        for block in content:
            t = block.get("type")
            if t == "text":
                parts.append({"type":"text","text":block["text"]})
            elif t == "image_url":
                url = block["image_url"]["url"]
                if url.startswith("data:"):
                    header, data = url.split(",",1)
                    mt = header.split(":")[1].split(";")[0]
                    parts.append({"type":"image","source":{"type":"base64","media_type":mt,"data":data}})
                else:
                    parts.append({"type":"image","source":{"type":"url","url":url}})
        return parts

    @staticmethod
    def _ant_content_to_oai_parts(blocks) -> List[dict]:
        if isinstance(blocks, str):
            return [{"type":"text","text":blocks}]
        parts = []
        for b in blocks:
            bt = b.get("type")
            if bt == "text":
                parts.append({"type":"text","text":b["text"]})
            elif bt == "image":
                src = b["source"]
                url = f"data:{src['media_type']};base64,{src['data']}" if src["type"]=="base64" else src["url"]
                parts.append({"type":"image_url","image_url":{"url":url}})
            elif bt == "thinking":
                parts.append({"type":"text","text":f"<thinking>{b.get('thinking','')}</thinking>"})
        return parts

    @staticmethod
    def _parse_args(raw) -> dict:
        if isinstance(raw, str):
            try: return json.loads(raw)
            except: return {}
        return raw or {}

    # ── OpenAI → Anthropic ───────────────────────────────────────

    @staticmethod
    def openai_to_anthropic(body: dict) -> dict:
        out: dict = {}
        system_parts: List[dict] = []
        messages_out: List[dict] = []

        for msg in body.get("messages", []):
            role    = msg.get("role","")
            content = msg.get("content")

            if role in ("system","developer"):
                text = content if isinstance(content,str) \
                    else " ".join(b.get("text","") for b in (content or []) if isinstance(b,dict))
                system_parts.append({"type":"text","text":text})

            elif role == "user":
                messages_out.append({"role":"user","content":ReqConv._oai_content_to_anthropic_parts(content or "")})

            elif role == "assistant":
                ant_content: List[dict] = []
                if isinstance(content, str) and content:
                    ant_content.append({"type":"text","text":content})
                elif isinstance(content, list):
                    for b in content:
                        if b.get("type")=="text": ant_content.append({"type":"text","text":b["text"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function",{})
                    ant_content.append({"type":"tool_use","id":tc["id"],"name":fn.get("name",""),
                                        "input":ReqConv._parse_args(fn.get("arguments",{}))})
                if ant_content:
                    messages_out.append({"role":"assistant","content":ant_content})

            elif role == "tool":
                tr = {"type":"tool_result","tool_use_id":msg.get("tool_call_id",""),"content":msg.get("content","")}
                if (messages_out and messages_out[-1]["role"]=="user"
                        and isinstance(messages_out[-1]["content"],list)
                        and any(b.get("type")=="tool_result" for b in messages_out[-1]["content"])):
                    messages_out[-1]["content"].append(tr)
                else:
                    messages_out.append({"role":"user","content":[tr]})

        if system_parts:
            out["system"] = system_parts[0]["text"] if len(system_parts)==1 else system_parts
        out["messages"] = messages_out
        out["model"]    = body.get("model","")
        mt = body.get("max_tokens") or body.get("max_completion_tokens")
        out["max_tokens"] = mt or 4096
        for key in ("temperature","top_p","stream","metadata"):
            if key in body: out[key] = body[key]
        if "stop" in body:
            v = body["stop"]
            out["stop_sequences"] = [v] if isinstance(v,str) else v

        oai_tools = body.get("tools") or []
        if oai_tools:
            out["tools"] = [{"name":t["function"].get("name",""),"description":t["function"].get("description",""),
                             "input_schema":t["function"].get("parameters",{"type":"object","properties":{}})}
                            for t in oai_tools if t.get("type")=="function"]

        tc = body.get("tool_choice")
        if tc:
            if   tc == "auto":     out["tool_choice"] = {"type":"auto"}
            elif tc == "required": out["tool_choice"] = {"type":"any"}
            elif tc == "none":     out["tool_choice"] = {"type":"none"}
            elif isinstance(tc,dict) and tc.get("type")=="function":
                out["tool_choice"] = {"type":"tool","name":tc["function"]["name"]}
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

        system = body.get("system")
        if system:
            text = system if isinstance(system,str) \
                else " ".join(b.get("text","") for b in system if isinstance(b,dict))
            messages_out.append({"role":"system","content":text})

        for msg in body.get("messages",[]):
            role    = msg["role"]
            content = msg["content"]
            if role == "user":
                if isinstance(content,str):
                    messages_out.append({"role":"user","content":content})
                elif isinstance(content,list):
                    trs     = [b for b in content if b.get("type")=="tool_result"]
                    normal  = [b for b in content if b.get("type")!="tool_result"]
                    for tr in trs:
                        tc_content = tr.get("content","")
                        messages_out.append({"role":"tool","tool_call_id":tr.get("tool_use_id",""),
                                             "content":tc_content if isinstance(tc_content,str) else json.dumps(tc_content)})
                    if normal:
                        parts = ReqConv._ant_content_to_oai_parts(normal)
                        messages_out.append({"role":"user",
                                             "content":parts[0]["text"] if len(parts)==1 and parts[0]["type"]=="text" else parts})
            elif role == "assistant":
                blocks    = content if isinstance(content,list) else [{"type":"text","text":content}]
                text_parts= [b for b in blocks if b.get("type")=="text"]
                tool_uses = [b for b in blocks if b.get("type")=="tool_use"]
                think_pts = [b for b in blocks if b.get("type")=="thinking"]
                text_str  = "".join(b["text"] for b in text_parts)
                for th in think_pts: text_str += f"\n<thinking>{th.get('thinking','')}</thinking>"
                if tool_uses:
                    messages_out.append({"role":"assistant","content":text_str or None,
                        "tool_calls":[{"id":tu["id"],"type":"function","function":{"name":tu["name"],"arguments":json.dumps(tu.get("input",{}))}} for tu in tool_uses]})
                else:
                    messages_out.append({"role":"assistant","content":text_str})

        out["messages"] = messages_out
        out["model"]    = body.get("model","")
        if body.get("max_tokens"): out["max_tokens"] = body["max_tokens"]
        for key in ("temperature","top_p","stream"):
            if key in body: out[key] = body[key]
        if "stop_sequences" in body: out["stop"] = body["stop_sequences"]
        if body.get("tools"):
            out["tools"] = [{"type":"function","function":{"name":t.get("name",""),"description":t.get("description",""),
                             "parameters":t.get("input_schema",{"type":"object","properties":{}})}} for t in body["tools"]]
        tc = body.get("tool_choice")
        if tc:
            tt = tc.get("type","auto")
            if   tt=="auto": out["tool_choice"]="auto"
            elif tt=="none": out["tool_choice"]="none"
            elif tt=="any":  out["tool_choice"]="required"
            elif tt=="tool": out["tool_choice"]={"type":"function","function":{"name":tc.get("name","")}}
        for f in ("thinking","betas","system","metadata","stop_sequences","top_k","anthropic_version"):
            out.pop(f,None)
        return out

    # ── OpenAI → Gemini ──────────────────────────────────────────

    @staticmethod
    def openai_to_gemini(body: dict) -> Tuple[dict, str]:
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
                if isinstance(content,str): parts=[{"text":content}]
                else:
                    parts=[]
                    for b in (content or []):
                        if b.get("type")=="text": parts.append({"text":b["text"]})
                        elif b.get("type")=="image_url":
                            url=b["image_url"]["url"]
                            if url.startswith("data:"):
                                hdr,data=url.split(",",1); mt=hdr.split(":")[1].split(";")[0]
                                parts.append({"inline_data":{"mime_type":mt,"data":data}})
                            else:
                                parts.append({"file_data":{"file_uri":url,"mime_type":"image/jpeg"}})
                contents.append({"role":"user","parts":parts})
            elif role == "assistant":
                parts=[]
                if content:
                    txt=content if isinstance(content,str) else "".join(b.get("text","") for b in content if isinstance(b,dict) and b.get("type")=="text")
                    if txt: parts.append({"text":txt})
                for tc in msg.get("tool_calls") or []:
                    fn=tc.get("function",{}); parts.append({"functionCall":{"name":fn.get("name",""),"args":ReqConv._parse_args(fn.get("arguments",{}))}})
                if parts: contents.append({"role":"model","parts":parts})
            elif role == "tool":
                fr={"name":msg.get("name",msg.get("tool_call_id","")),"response":{"content":msg.get("content","")}}
                if (contents and contents[-1]["role"]=="user" and any("functionResponse" in p for p in contents[-1]["parts"])):
                    contents[-1]["parts"].append({"functionResponse":fr})
                else:
                    contents.append({"role":"user","parts":[{"functionResponse":fr}]})

        out["contents"] = contents
        if system_instruction: out["system_instruction"] = system_instruction

        oai_tools = body.get("tools") or []
        if oai_tools:
            fn_decls=[{"name":t["function"].get("name",""),"description":t["function"].get("description",""),
                       "parameters":t["function"].get("parameters",{})} for t in oai_tools if t.get("type")=="function"]
            out["tools"]=[{"functionDeclarations":fn_decls}]

        tc=body.get("tool_choice")
        if tc:
            if   tc=="none":     out["tool_config"]={"function_calling_config":{"mode":"NONE"}}
            elif tc=="required": out["tool_config"]={"function_calling_config":{"mode":"ANY"}}
            elif isinstance(tc,dict) and tc.get("type")=="function":
                out["tool_config"]={"function_calling_config":{"mode":"ANY","allowed_function_names":[tc["function"]["name"]]}}

        gc: dict={}
        mt=body.get("max_tokens") or body.get("max_completion_tokens")
        if mt: gc["maxOutputTokens"]=mt
        if "temperature" in body: gc["temperature"]=body["temperature"]
        if "top_p"       in body: gc["topP"]=body["top_p"]
        stop=body.get("stop")
        if stop: gc["stopSequences"]=[stop] if isinstance(stop,str) else stop
        if body.get("response_format",{}).get("type")=="json_object": gc["responseMimeType"]="application/json"
        if gc: out["generation_config"]=gc
        return out, model

    # ── Gemini → OpenAI ──────────────────────────────────────────

    @staticmethod
    def gemini_to_openai(body: dict) -> dict:
        out: dict = {}
        messages: List[dict] = []
        si=body.get("system_instruction")
        if si:
            messages.append({"role":"system","content":" ".join(p.get("text","") for p in si.get("parts",[]))})
        for c in body.get("contents",[]):
            role=c.get("role","user"); oai_role="assistant" if role=="model" else "user"
            parts=c.get("parts",[]); texts=[p["text"] for p in parts if "text" in p]
            fn_calls=[p["functionCall"] for p in parts if "functionCall" in p]
            fn_resp=[p["functionResponse"] for p in parts if "functionResponse" in p]
            if fn_resp:
                for fr in fn_resp:
                    messages.append({"role":"tool","tool_call_id":fr["name"],"name":fr["name"],"content":json.dumps(fr.get("response",{}))})
            elif fn_calls:
                messages.append({"role":oai_role,"content":" ".join(texts) or None,
                    "tool_calls":[{"id":fc["name"],"type":"function","function":{"name":fc["name"],"arguments":json.dumps(fc.get("args",{}))}} for fc in fn_calls]})
            else:
                messages.append({"role":oai_role,"content":" ".join(texts)})
        out["messages"]=messages
        for tb in body.get("tools",[]):
            fn_decls=tb.get("functionDeclarations",[])
            if fn_decls:
                out["tools"]=[{"type":"function","function":{"name":fd["name"],"description":fd.get("description",""),"parameters":fd.get("parameters",{})}} for fd in fn_decls]
        gc=body.get("generation_config",{})
        if "maxOutputTokens" in gc: out["max_tokens"]=gc["maxOutputTokens"]
        if "temperature"     in gc: out["temperature"]=gc["temperature"]
        if "topP"            in gc: out["top_p"]=gc["topP"]
        if "stopSequences"   in gc: out["stop"]=gc["stopSequences"]
        return out

    @staticmethod
    def anthropic_to_gemini(body:dict) -> Tuple[dict,str]:
        return ReqConv.openai_to_gemini(ReqConv.anthropic_to_openai(body))

    @staticmethod
    def gemini_to_anthropic(body:dict) -> dict:
        return ReqConv.openai_to_anthropic(ReqConv.gemini_to_openai(body))


def convert_request(body:dict, from_fmt:str, to_fmt:str) -> Tuple[dict,str]:
    if from_fmt == to_fmt: return body, body.get("model","")
    if from_fmt=="openai"    and to_fmt=="anthropic": return ReqConv.openai_to_anthropic(body),""
    if from_fmt=="anthropic" and to_fmt=="openai":    return ReqConv.anthropic_to_openai(body),""
    if from_fmt=="openai"    and to_fmt=="gemini":
        b,m=ReqConv.openai_to_gemini(body); return b,m
    if from_fmt=="gemini"    and to_fmt=="openai":    return ReqConv.gemini_to_openai(body),""
    if from_fmt=="anthropic" and to_fmt=="gemini":
        b,m=ReqConv.anthropic_to_gemini(body); return b,m
    if from_fmt=="gemini"    and to_fmt=="anthropic": return ReqConv.gemini_to_anthropic(body),""
    return body, body.get("model","")

# ─────────────────────────────────────────────────────────────────
# RESPONSE CONVERTERS
# ─────────────────────────────────────────────────────────────────

class RespConv:
    @staticmethod
    def anthropic_to_openai(r:dict) -> dict:
        blocks=r.get("content",[])
        texts=[b["text"] for b in blocks if b.get("type")=="text"]
        tuses=[b for b in blocks if b.get("type")=="tool_use"]
        fm={"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls","stop_sequence":"stop"}
        finish=fm.get(r.get("stop_reason","end_turn"),"stop")
        if tuses:
            msg={"role":"assistant","content":"".join(texts) or None,"refusal":None,
                 "tool_calls":[{"id":tu["id"],"type":"function","function":{"name":tu["name"],"arguments":json.dumps(tu.get("input",{}))}} for tu in tuses]}
        else:
            msg={"role":"assistant","content":"".join(texts),"refusal":None}
        u=r.get("usage",{})
        return {"id":f"chatcmpl-{r.get('id','')}","object":"chat.completion","created":int(time.time()),
                "model":r.get("model",""),"choices":[{"index":0,"message":msg,"finish_reason":finish}],
                "usage":{"prompt_tokens":u.get("input_tokens",0),"completion_tokens":u.get("output_tokens",0),
                         "total_tokens":u.get("input_tokens",0)+u.get("output_tokens",0)}}

    @staticmethod
    def gemini_to_openai(r:dict, model:str) -> dict:
        choices=[]
        for i,cand in enumerate(r.get("candidates",[])):
            parts=cand.get("content",{}).get("parts",[])
            texts=[p["text"] for p in parts if "text" in p and not p.get("thought")]
            fn_calls=[p["functionCall"] for p in parts if "functionCall" in p]
            if fn_calls:
                msg={"role":"assistant","content":None,"tool_calls":[{"id":fc["name"],"type":"function","function":{"name":fc["name"],"arguments":json.dumps(fc.get("args",{}))}} for fc in fn_calls]}
                finish="tool_calls"
            else:
                msg={"role":"assistant","content":"".join(texts)}
                fm={"STOP":"stop","MAX_TOKENS":"length","SAFETY":"content_filter"}
                finish=fm.get(cand.get("finishReason","STOP"),"stop")
            choices.append({"index":i,"message":msg,"finish_reason":finish})
        u=r.get("usageMetadata",{})
        return {"id":f"chatcmpl-gemini-{int(time.time())}","object":"chat.completion","created":int(time.time()),"model":model,
                "choices":choices,"usage":{"prompt_tokens":u.get("promptTokenCount",0),"completion_tokens":u.get("candidatesTokenCount",0),"total_tokens":u.get("totalTokenCount",0)}}

    @staticmethod
    def openai_to_anthropic(r:dict) -> dict:
        choices=r.get("choices",[])
        content:List[dict]=[]; stop_reason="end_turn"
        if choices:
            msg=choices[0].get("message",{})
            if msg.get("content"): content.append({"type":"text","text":msg["content"]})
            for tc in msg.get("tool_calls",[]) or []:
                fn=tc.get("function",{})
                content.append({"type":"tool_use","id":tc["id"],"name":fn.get("name",""),
                    "input":json.loads(fn["arguments"]) if isinstance(fn.get("arguments"),str) else fn.get("arguments",{})})
            fm={"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use","content_filter":"end_turn"}
            stop_reason=fm.get(choices[0].get("finish_reason","stop"),"end_turn")
        u=r.get("usage",{})
        return {"id":f"msg_{r.get('id','')}","type":"message","role":"assistant","content":content,
                "model":r.get("model",""),"stop_reason":stop_reason,"stop_sequence":None,
                "usage":{"input_tokens":u.get("prompt_tokens",0),"output_tokens":u.get("completion_tokens",0)}}

    @staticmethod
    def gemini_to_anthropic(r:dict, model:str) -> dict:
        return RespConv.openai_to_anthropic(RespConv.gemini_to_openai(r,model))

    @staticmethod
    def openai_to_gemini(r:dict) -> dict:
        candidates=[]
        for i,ch in enumerate(r.get("choices",[])):
            msg=ch.get("message",{}); parts=[]
            if msg.get("content"): parts.append({"text":msg["content"]})
            for tc in msg.get("tool_calls",[]) or []:
                fn=tc.get("function",{}); args=json.loads(fn["arguments"]) if isinstance(fn.get("arguments"),str) else fn.get("arguments",{})
                parts.append({"functionCall":{"name":fn.get("name",""),"args":args}})
            fm={"stop":"STOP","length":"MAX_TOKENS","tool_calls":"STOP","content_filter":"SAFETY"}
            candidates.append({"content":{"parts":parts,"role":"model"},"finishReason":fm.get(ch.get("finish_reason","stop"),"STOP"),"index":i})
        u=r.get("usage",{})
        return {"candidates":candidates,"usageMetadata":{"promptTokenCount":u.get("prompt_tokens",0),"candidatesTokenCount":u.get("completion_tokens",0),"totalTokenCount":u.get("total_tokens",0)}}

    @staticmethod
    def anthropic_to_gemini(r:dict) -> dict:
        return RespConv.openai_to_gemini(RespConv.anthropic_to_openai(r))

    @staticmethod
    def convert(resp:dict, provider_fmt:str, client_fmt:str, model:str="") -> dict:
        if provider_fmt == client_fmt: return resp
        if provider_fmt=="anthropic": oai=RespConv.anthropic_to_openai(resp)
        elif provider_fmt=="gemini":  oai=RespConv.gemini_to_openai(resp,model)
        else:                         oai=resp
        if client_fmt=="openai":    return oai
        if client_fmt=="anthropic": return RespConv.openai_to_anthropic(oai)
        if client_fmt=="gemini":    return RespConv.openai_to_gemini(oai)
        return oai

# ─────────────────────────────────────────────────────────────────
# STREAMING CONVERSION
# ─────────────────────────────────────────────────────────────────

def _parse_sse(raw:str) -> List[Tuple[str,Any]]:
    events: List[Tuple[str,Any]]=[]
    current_event="message"
    for line in raw.splitlines():
        if   line.startswith("event:"): current_event=line[6:].strip()
        elif line.startswith("data:"):
            ds=line[5:].strip()
            if ds=="[DONE]": events.append(("done",None)); current_event="message"
            else:
                try: events.append((current_event,json.loads(ds))); current_event="message"
                except: pass
        elif line=="": current_event="message"
    return events


def _reconstruct_from_events(events,provider_fmt):
    text=""; tool_calls: Dict[int,dict]={}; stop_reason="end_turn"
    for ev_name,data in events:
        if data is None: continue
        if provider_fmt=="openai":
            for ch in data.get("choices",[]):
                delta=ch.get("delta",{}); finish=ch.get("finish_reason")
                if delta.get("content"): text+=delta["content"]
                for tc_delta in delta.get("tool_calls",[]) or []:
                    idx=tc_delta.get("index",0)
                    if idx not in tool_calls: tool_calls[idx]={"id":"","name":"","args":""}
                    if tc_delta.get("id"): tool_calls[idx]["id"]=tc_delta["id"]
                    fn=tc_delta.get("function",{})
                    if fn.get("name"):      tool_calls[idx]["name"]+=fn["name"]
                    if fn.get("arguments"): tool_calls[idx]["args"]+=fn["arguments"]
                if finish:
                    fm={"stop":"end_turn","length":"max_tokens","tool_calls":"tool_use"}
                    stop_reason=fm.get(finish,"end_turn")
        elif provider_fmt=="anthropic":
            if ev_name=="content_block_delta":
                d=data.get("delta",{}); idx=data.get("index",0)
                if d.get("type")=="text_delta": text+=d.get("text","")
                elif d.get("type")=="input_json_delta":
                    if idx not in tool_calls: tool_calls[idx]={"id":"","name":"","args":""}
                    tool_calls[idx]["args"]+=d.get("partial_json","")
            elif ev_name=="content_block_start":
                cb=data.get("content_block",{})
                if cb.get("type")=="tool_use":
                    idx=data.get("index",0); tool_calls[idx]={"id":cb.get("id",""),"name":cb.get("name",""),"args":""}
            elif ev_name=="message_delta":
                stop_reason=data.get("delta",{}).get("stop_reason","end_turn") or "end_turn"
        elif provider_fmt=="gemini":
            for cand in data.get("candidates",[]):
                for part in cand.get("content",{}).get("parts",[]):
                    if "text" in part and not part.get("thought"): text+=part["text"]
                    elif "functionCall" in part:
                        fc=part["functionCall"]; idx=len(tool_calls)
                        tool_calls[idx]={"id":fc["name"],"name":fc["name"],"args":json.dumps(fc.get("args",{}))}
                if cand.get("finishReason")=="STOP": stop_reason="end_turn"
    tc_list=[{"id":v["id"] or f"tc_{k}","type":"function","function":{"name":v["name"],"arguments":v["args"]}} for k,v in sorted(tool_calls.items())]
    return text,tc_list,stop_reason


def _emit_openai_stream(text,tool_calls,stop_reason,model):
    mid=f"chatcmpl-proxy-{int(time.time())}"; out=""
    out+=f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"
    if text: out+=f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'content':text},'finish_reason':None}]})}\n\n"
    for i,tc in enumerate(tool_calls):
        out+=f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':i,'id':tc['id'],'type':'function','function':{'name':tc['function']['name'],'arguments':''}}]},'finish_reason':None}]})}\n\n"
        out+=f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{'tool_calls':[{'index':i,'function':{'arguments':tc['function']['arguments']}}]},'finish_reason':None}]})}\n\n"
    fm={"end_turn":"stop","max_tokens":"length","tool_use":"tool_calls"}
    out+=f"data: {json.dumps({'id':mid,'object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':fm.get(stop_reason,'stop')}]})}\n\n"
    out+="data: [DONE]\n\n"
    return out


def _emit_anthropic_stream(text,tool_calls,stop_reason,model):
    mid=f"msg_proxy_{int(time.time())}"; out=""
    out+=f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':mid,'type':'message','role':'assistant','content':[],'model':model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    idx=0
    if text:
        out+=f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'text','text':''}})}\n\n"
        out+=f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'text_delta','text':text}})}\n\n"
        out+=f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"
        idx+=1
    for tc in tool_calls:
        fn=tc["function"]
        out+=f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':tc['id'],'name':fn['name'],'input':{}}})}\n\n"
        out+=f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':fn['arguments']}})}\n\n"
        out+=f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"
        idx+=1
    out+=f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    out+=f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
    return out


def _emit_gemini_stream(text,tool_calls,stop_reason,model):
    parts=[]
    if text: parts.append({"text":text})
    for tc in tool_calls:
        fn=tc["function"]; args=json.loads(fn["arguments"]) if isinstance(fn["arguments"],str) else fn["arguments"]
        parts.append({"functionCall":{"name":fn["name"],"args":args}})
    return f"data: {json.dumps({'candidates':[{'content':{'parts':parts,'role':'model'},'finishReason':'STOP','index':0}]})}\n\n"


def convert_stream(raw_sse:str, provider_fmt:str, client_fmt:str, model:str) -> str:
    events=_parse_sse(raw_sse)
    text,tool_calls,stop_reason=_reconstruct_from_events(events,provider_fmt)
    if   client_fmt=="openai":    return _emit_openai_stream(text,tool_calls,stop_reason,model)
    elif client_fmt=="anthropic": return _emit_anthropic_stream(text,tool_calls,stop_reason,model)
    elif client_fmt=="gemini":    return _emit_gemini_stream(text,tool_calls,stop_reason,model)
    return raw_sse

# ─────────────────────────────────────────────────────────────────
# URL BUILDER
# ─────────────────────────────────────────────────────────────────

def build_url(provider:dict, client_path:str, provider_fmt:str, model:str, is_stream:bool) -> str:
    base=provider["base_url"].rstrip("/")
    if provider_fmt=="gemini":
        action="streamGenerateContent" if is_stream else "generateContent"
        m=model or "gemini-2.0-flash"
        url=f"{base}/v1beta/models/{m}:{action}"
        if is_stream: url+="?alt=sse"
        return url
    if provider_fmt=="anthropic": return f"{base}/v1/messages"
    pl=client_path.split("?")[0].lower()
    if "/v1/chat/completions" in pl: return f"{base}/v1/chat/completions"
    if "/api/chat"            in pl: return f"{base}/api/chat"
    if "/v1/completions"      in pl: return f"{base}/v1/completions"
    if "/v1/responses"        in pl: return f"{base}/v1/chat/completions"
    return f"{base}/v1/chat/completions"

# ─────────────────────────────────────────────────────────────────
# HEADER BUILDER  ← injects API key OR OAuth Bearer token
# ─────────────────────────────────────────────────────────────────

def build_headers(provider:dict, cred:"Credential", incoming_headers, body:dict) -> dict:
    """
    Build outgoing headers for the upstream provider.
    Handles three auth types:
      api_key  → classic key placement (x-api-key / Bearer / x-goog-api-key)
      oauth    → Bearer token (used like api_key but value is access_token from file)
      bearer   → same as oauth; raw bearer value from config
    """
    fmt = provider["native_format"]
    hdrs: dict = {}

    if body:
        hdrs["Content-Type"] = "application/json"

    token = cred.token

    if fmt == "anthropic":
        if cred.cred_type == "api_key":
            hdrs["x-api-key"] = token
        else:
            # OAuth access_token goes as Bearer
            hdrs["Authorization"] = f"Bearer {token}"
        hdrs["anthropic-version"] = "2023-06-01"
        # Claude Code OAuth also needs these to look like the real CLI
        if cred.cred_type in ("oauth","bearer"):
            hdrs["anthropic-beta"] = "claude-code-20250219"
            hdrs.setdefault("User-Agent", "claude-code/1.0.0")

    elif fmt == "gemini":
        if cred.cred_type == "api_key":
            hdrs["x-goog-api-key"] = token
        else:
            hdrs["Authorization"] = f"Bearer {token}"

    else:  # openai-compatible (Codex, Groq, OpenRouter, Ollama, Together…)
        hdrs["Authorization"] = f"Bearer {token}"

    hdrs["Host"] = urllib.parse.urlparse(provider["base_url"]).netloc

    # Extra headers from config
    hdrs.update(provider.get("extra_headers", {}))

    # Forward safe client headers (skip auth/connection control)
    skip = {"host","authorization","x-api-key","x-goog-api-key","content-length",
            "content-type","connection","keep-alive","transfer-encoding"}
    for k,v in incoming_headers.items():
        if k.lower() not in skip:
            hdrs[k] = v

    return hdrs

# ─────────────────────────────────────────────────────────────────
# PROXY HANDLER
# ─────────────────────────────────────────────────────────────────

_cfg:  Optional[Config]             = None
_args: Optional[argparse.Namespace] = None

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):    self._passthrough()
    def do_DELETE(self): self._passthrough()
    def do_POST(self):   self._llm_request()
    def do_PUT(self):    self._passthrough()

    # ── GET/DELETE passthrough ───────────────────────────────────

    def _passthrough(self):
        provider=self._get_provider()
        if not provider: return
        mgr=get_cred_mgr(provider); cred=mgr.next_credential()
        if not cred: self._error(429,"All credentials on cooldown"); return
        target=f"{provider['base_url'].rstrip('/')}{self.path}"
        hdrs=build_headers(provider,cred,dict(self.headers),{})
        req=urllib.request.Request(target,headers=hdrs,method=self.command)
        try:
            with urllib.request.urlopen(req,timeout=_cfg.timeout) as r:
                self.send_response(r.getcode())
                for h,v in r.getheaders():
                    if h.lower() not in ("connection","transfer-encoding"): self.send_header(h,v)
                self.end_headers(); self.wfile.write(r.read())
        except Exception as e:
            self._error(502,str(e))

    # ── main LLM handler ─────────────────────────────────────────

    def _llm_request(self):
        ts=datetime.now().strftime("%H:%M:%S")

        try:
            length=int(self.headers.get("Content-Length",0))
            raw_body=self.rfile.read(length) if length>0 else b"{}"
            body=json.loads(raw_body) if raw_body.strip() else {}
        except Exception as e:
            self._error(400,f"Bad JSON: {e}"); return

        provider=self._get_provider()
        if not provider: return

        provider_fmt=provider["native_format"]
        client_fmt=_args.from_format if _args.from_format!="auto" \
                   else detect_client_format(self.path,dict(self.headers),body)
        reply_fmt=_args.to_format or client_fmt

        orig_model=body.get("model","")
        resolved_model=resolve_model(orig_model,provider)
        is_stream=bool(body.get("stream",False))

        if orig_model!=resolved_model:
            body=copy.deepcopy(body); body["model"]=resolved_model

        log(f"\n[{ts}] {BOLD}POST{RESET} {self.path}")
        log(f"  {BLUE}Client fmt  :{RESET} {client_fmt.upper()}"+(f" (reply as {reply_fmt.upper()})" if reply_fmt!=client_fmt else ""))
        log(f"  {MAGENTA}Provider    :{RESET} {provider['name']} [{provider_fmt.upper()}]")
        log(f"  {CYAN}Model       :{RESET} {orig_model or '(none)'}"+(f" → {GREEN}{resolved_model}{RESET}" if orig_model!=resolved_model else ""))
        log(f"  {GREY}Stream      : {is_stream}{RESET}")

        try:
            converted,gemini_model=convert_request(body,client_fmt,provider_fmt)
            gemini_model=gemini_model or resolved_model
        except Exception as e:
            self._error(500,f"Request conversion failed: {e}\n{traceback.format_exc()}"); return

        target_url=build_url(provider,self.path,provider_fmt,gemini_model,is_stream)
        vlog(f"  Target URL  : {target_url}")

        self._fire(provider=provider,converted=converted,target_url=target_url,
                   client_fmt=client_fmt,reply_fmt=reply_fmt,provider_fmt=provider_fmt,
                   is_stream=is_stream,model=gemini_model or resolved_model)

    # ── fire + retry ─────────────────────────────────────────────

    def _fire(self, provider, converted, target_url, client_fmt, reply_fmt,
              provider_fmt, is_stream, model):
        mgr=get_cred_mgr(provider); retries=_cfg.max_retries

        for attempt in range(retries):
            cred=mgr.next_credential()
            if cred is None:
                self._error(429,"All credentials on cooldown."); return

            # Label for logging
            if cred.cred_type in ("oauth","bearer"):
                label=f"OAuth({os.path.basename(cred.source)[:20]})"
            else:
                label=f"key …{cred.token[-10:]}"
            log(f"  {GREEN}[{attempt+1}/{retries}]{RESET} {label} [{cred.cred_type}]")

            headers=build_headers(provider,cred,self.headers,converted)
            payload=json.dumps(converted).encode("utf-8")
            req=urllib.request.Request(url=target_url,data=payload,headers=headers,method="POST")

            try:
                resp=urllib.request.urlopen(req,timeout=_cfg.timeout)
                log(f"  {GREEN}✓ {resp.getcode()}{RESET}")
                self._forward(resp,client_fmt,reply_fmt,provider_fmt,is_stream,model)
                return

            except urllib.error.HTTPError as e:
                log(f"  {YELLOW}⚠ HTTP {e.code} (attempt {attempt+1}/{retries}){RESET}")
                if e.code==429:
                    try: ra=int(e.headers.get("Retry-After",60))
                    except: ra=60
                    cred.rate_limit(ra)
                elif e.code in (401,403):
                    cred.invalidate()
                elif e.code not in _cfg.retry_codes:
                    self.send_response(e.code)
                    for h,v in e.headers.items():
                        if h.lower() not in ("connection","transfer-encoding"): self.send_header(h,v)
                    self.end_headers(); self.wfile.write(e.read())
                    return

            except Exception as e:
                log(f"  {RED}✗ {type(e).__name__}: {e}{RESET}")
                if attempt==retries-1:
                    self._error(502,f"All retries failed: {e}"); return

        self._error(502,f"All {retries} attempts failed.")

    # ── response forwarding ──────────────────────────────────────

    def _forward(self,resp,client_fmt,reply_fmt,provider_fmt,is_stream,model):
        if is_stream: self._forward_stream(resp,client_fmt,reply_fmt,provider_fmt,model)
        else:         self._forward_json(resp,client_fmt,reply_fmt,provider_fmt,model)

    def _forward_json(self,resp,client_fmt,reply_fmt,provider_fmt,model):
        raw=resp.read()
        try: body=json.loads(raw)
        except:
            self.send_response(200)
            for h,v in resp.getheaders():
                if h.lower() not in ("connection","transfer-encoding","content-length"): self.send_header(h,v)
            self.end_headers(); self.wfile.write(raw); return
        if provider_fmt!=reply_fmt:
            try: body=RespConv.convert(body,provider_fmt,reply_fmt,model)
            except Exception as e: log(f"  {YELLOW}⚠ Response conversion error: {e}{RESET}")
        out=json.dumps(body).encode("utf-8")
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)

    def _forward_stream(self,resp,client_fmt,reply_fmt,provider_fmt,model):
        same=(provider_fmt==reply_fmt)
        self.send_response(200); self.send_header("Content-Type","text/event-stream")
        self.send_header("Cache-Control","no-cache"); self.send_header("X-Accel-Buffering","no")
        if same:
            self.send_header("Transfer-Encoding","chunked"); self.end_headers()
            try:
                while True:
                    chunk=resp.read(4096)
                    if not chunk: break
                    self.wfile.write(chunk); self.wfile.flush()
            except: pass
            return
        buf=b""
        try:
            while True:
                chunk=resp.read(4096)
                if not chunk: break
                buf+=chunk
        except Exception as e: log(f"  {YELLOW}⚠ Stream read: {e}{RESET}")
        raw_text=buf.decode("utf-8",errors="replace")
        converted_sse=convert_stream(raw_text,provider_fmt,reply_fmt,model)
        out=converted_sse.encode("utf-8")
        self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out); self.wfile.flush()

    # ── helpers ──────────────────────────────────────────────────

    def _get_provider(self) -> Optional[dict]:
        name=_args.provider or _cfg.default_provider
        p=_cfg.get(name)
        if not p:
            self._error(400,f"Provider '{name}' not found/enabled. Available: {list(_cfg.providers.keys())}")
            return None
        return p

    def _error(self,code:int,msg:str):
        log(f"  {RED}✗ {code} — {msg}{RESET}")
        body=json.dumps({"error":{"message":msg,"type":"proxy_error","code":code}}).encode()
        try:
            self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        except: pass

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    global _cfg, _args, _verbose
    _args=parse_args(); _verbose=_args.verbose; _cfg=Config(_args.config)
    host=_args.host or _cfg.server.get("host","localhost")
    port=_args.port or _cfg.server.get("port",8000)
    pname=_args.provider or _cfg.default_provider
    if pname not in _cfg.providers:
        log(f"\n{RED}✗ Provider '{pname}' not enabled.{RESET}")
        log(f"  Available: {list(_cfg.providers.keys())}"); sys.exit(1)

    provider=_cfg.providers[pname]
    native_fmt=provider["native_format"]
    mgr=get_cred_mgr(provider)
    reply_label=f"→ {_args.to_format.upper()}" if _args.to_format else "(mirrors client)"

    # Credential summary
    cred_parts=[]
    if provider.get("api_keys"):               cred_parts.append(f"{len(provider['api_keys'])} api_key(s)")
    if provider.get("oauth_token_files"):      cred_parts.append(f"{len(provider['oauth_token_files'])} oauth_file(s)")
    if provider.get("oauth_bearer_tokens"):    cred_parts.append(f"{len(provider['oauth_bearer_tokens'])} bearer(s)")
    cred_summary = " + ".join(cred_parts) if cred_parts else "⚠ NONE configured"

    print(f"""
{BOLD}{'═'*64}{RESET}
{BOLD}  🔀  Universal LLM Proxy  (API-key + OAuth edition){RESET}
{'═'*64}
  Listen     : http://{host}:{port}
  Provider   : {CYAN}{pname}{RESET}  [{native_fmt.upper()} format]
  Target     : {provider['base_url']}
  From       : {_args.from_format.upper()}  (auto-detect if 'auto')
  To (reply) : {reply_label}
  Credentials: {cred_summary}
  Retry codes: {_cfg.retry_codes}
  Max retries: {_cfg.max_retries}
{'═'*64}
  Auth type legend:
    api_key   → plain API key (x-api-key / Bearer / x-goog-api-key)
    oauth     → JSON token file with access_token + auto-refresh
    bearer    → raw bearer token (no refresh; e.g. iFlow cookie token)
{'═'*64}
""")

    socketserver.TCPServer.allow_reuse_address=True
    with socketserver.ThreadingTCPServer((host,port),ProxyHandler) as srv:
        try: srv.serve_forever()
        except KeyboardInterrupt: print(f"\n{YELLOW}  ⏹  Proxy stopped.{RESET}\n")

if __name__=="__main__":
    main()

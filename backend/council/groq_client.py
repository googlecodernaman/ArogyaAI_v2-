"""
LLM Council — multi-provider client.

Council roster:
  Member A  — Groq   llama-3.3-70b-versatile   (fast 70B, deep reasoning)
  Member B  — DeepSeek  deepseek-chat           (strong clinical logic)
  Member C  — Mistral   mistral-small-latest    (diverse perspective)
  Chairman  — Groq   llama-3.3-70b-versatile   (synthesis + final answer)
  Reviewer  — Groq   llama-3.1-8b-instant      (fast convergence ranking)

Transport:
  Groq   → official AsyncGroq SDK
  Others → direct httpx POST to OpenAI-compatible /v1/chat/completions
           (no Gemini SDK, no Mistral SDK — avoids broken installs on Windows)
"""

from __future__ import annotations

import asyncio
import json

import httpx
from groq import AsyncGroq

from backend.config import get_settings

settings = get_settings()

# ── Model roster ──────────────────────────────────────────────────────────────

COUNCIL_MODELS = {
    "member_a": {"model": "llama-3.3-70b-versatile", "provider": "groq"},
    "member_b": {"model": "deepseek-chat",            "provider": "deepseek"},
    "member_c": {"model": "mistral-small-latest",     "provider": "mistral"},
    "chairman": {"model": "llama-3.3-70b-versatile", "provider": "groq"},
    "reviewer": {"model": "llama-3.1-8b-instant",    "provider": "groq"},
}

# Provider endpoint map (OpenAI-compatible)
_ENDPOINTS: dict[str, str] = {
    "deepseek":   "https://api.deepseek.com/v1/chat/completions",
    "mistral":    "https://api.mistral.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

_API_KEYS: dict[str, str] = {}   # populated lazily below

COUNCIL_SYSTEM = (
    "You are a clinical reasoning assistant. The patient case has been de-identified. "
    "Reply ONLY with a valid JSON object — no markdown fences, no text outside JSON. "
    'Keys: "differentials" (list of strings), "next_steps" (list of strings), '
    '"confidence" (float 0-1), "red_flag" (boolean).'
)

_FALLBACK = '{"differentials":[],"next_steps":[],"confidence":0,"red_flag":false}'


def _api_key(provider: str) -> str:
    """Lazily resolve API key so settings is read after .env is loaded."""
    mapping = {
        "deepseek":   settings.deepseek_api_key,
        "mistral":    settings.mistral_api_key,
        "openrouter": settings.openrouter_api_key,
    }
    return mapping.get(provider, "")


# ── Groq (SDK) ────────────────────────────────────────────────────────────────

async def _query_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    client = AsyncGroq(api_key=settings.groq_api_key)
    kwargs = {"reasoning_effort": "none"} if "qwen3" in model else {}
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content


# ── Generic OpenAI-compatible (httpx) ────────────────────────────────────────

async def _query_openai_compat(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Fire a plain POST to any OpenAI-compatible /v1/chat/completions endpoint.
    Works for DeepSeek, Mistral REST, OpenRouter, vLLM — zero SDK overhead.
    """
    url     = _ENDPOINTS[provider]
    api_key = _api_key(provider)
    if not api_key:
        raise ValueError(f"No API key configured for provider '{provider}'")

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        # OpenRouter requires a site header
        "HTTP-Referer":  "https://aarogyaai.local",
        "X-Title":       "AarogyaAI",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


# ── Public wrapper ────────────────────────────────────────────────────────────

async def query_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Query Groq directly (used by orchestrator for Chairman / Reviewer)."""
    return await _query_groq(model, messages, temperature, max_tokens)


async def _query_member(
    name: str,
    cfg: dict,
    system_msg: dict,
    user_msg: dict,
) -> tuple[str, str]:
    """Route a single council member to the correct provider."""
    try:
        provider = cfg["provider"]
        model    = cfg["model"]
        messages = [system_msg, user_msg]

        if provider == "groq":
            text = await _query_groq(model, messages, max_tokens=512)
        elif provider in _ENDPOINTS:
            text = await _query_openai_compat(provider, model, messages, max_tokens=512)
        else:
            print(f"[Council] {name}: unknown provider '{provider}', using fallback")
            text = _FALLBACK

    except Exception as e:
        print(f"[Council] {name} ({cfg['provider']}) failed: {type(e).__name__}: {e}")
        text = _FALLBACK

    return name, text


async def query_council_parallel(sanitized_prompt: str) -> dict[str, str]:
    """
    Stage 1 — Divergence: fan-out to Member A (Groq), B (DeepSeek), C (Mistral)
    in parallel. Returns {member_name: raw_response_text}.
    """
    system_msg = {"role": "system", "content": COUNCIL_SYSTEM}
    user_msg   = {"role": "user",   "content": sanitized_prompt}

    members = {k: v for k, v in COUNCIL_MODELS.items()
               if k in ("member_a", "member_b", "member_c")}

    results = await asyncio.gather(
        *[_query_member(n, c, system_msg, user_msg) for n, c in members.items()]
    )
    return dict(results)

"""
Groq API client for the LLM Council.
Uses asyncio.gather for parallel council member queries.

Model assignments (verified live on Groq as of Feb 2026):
  Member A  — llama-3.3-70b-versatile   (fast 70B general reasoning)
  Member B  — llama-3.1-8b-instant      (ultra-fast 8B, diverse perspective)
  Member C  — qwen/qwen3-32b            (Qwen3 32B — strong diverse reasoning)
  Chairman  — llama-3.3-70b-versatile    (synthesis + final answer)
  Reviewer  — llama-3.1-8b-instant       (fast peer ranking only)
"""
import asyncio
from groq import AsyncGroq
from config import get_settings

settings = get_settings()

# ── Model roster — 3 providers, 3 independent architectures ──────────────────
COUNCIL_MODELS = {
    "member_a": {"model": "llama-3.3-70b-versatile",  "provider": "groq"},     # Groq
    "member_b": {"model": "gemini-2.0-flash",          "provider": "gemini"},
    "member_c": {"model": "mistral-small-latest",      "provider": "mistral"},  # Mistral
    "chairman": {"model": "llama-3.3-70b-versatile",  "provider": "groq"},
    "reviewer": {"model": "llama-3.1-8b-instant",     "provider": "groq"},
}

# Shared system prompt for all council members
COUNCIL_SYSTEM = (
    "You are a clinical reasoning assistant. The patient case has been de-identified. "
    "Reply ONLY with a valid JSON object — no markdown fences, no text outside JSON. "
    'Keys: "differentials" (list of strings), "next_steps" (list of strings), '
    '"confidence" (float 0-1), "red_flag" (boolean).'
)

_FALLBACK = '{"differentials":[],"next_steps":[],"confidence":0,"red_flag":false}'


# ── Provider-specific query functions ─────────────────────────────────────────

async def _query_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    client = AsyncGroq(api_key=settings.groq_api_key)
    # Disable chain-of-thought for qwen3 so output is pure JSON (no <think> tags)
    kwargs = {"reasoning_effort": "none"} if "qwen3" in model else {}
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content


async def _query_gemini(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    from google import genai               # lazy import (google-genai SDK)
    from google.genai import types
    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


async def _query_mistral(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    from mistralai import Mistral  # lazy import
    client = Mistral(api_key=settings.mistral_api_key)
    resp = await client.chat.complete_async(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────────────────

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
    """Route a single council member query to the correct provider."""
    try:
        provider = cfg["provider"]
        model    = cfg["model"]
        if provider == "groq":
            text = await _query_groq(model, [system_msg, user_msg], max_tokens=512)
        elif provider == "gemini":
            text = await _query_gemini(model, COUNCIL_SYSTEM, user_msg["content"], max_tokens=512)
        elif provider == "mistral":
            text = await _query_mistral(model, [system_msg, user_msg], max_tokens=512)
        else:
            text = _FALLBACK
    except Exception as e:
        print(f"[Council] {name} ({cfg['provider']}) failed: {e}")
        text = _FALLBACK
    return name, text


async def query_council_parallel(sanitized_prompt: str) -> dict[str, str]:
    """
    Stage 1 — Divergence: fan-out to Members A (Groq), B (Gemini), C (Mistral)
    in parallel across 3 independent providers.
    Returns {member_name: raw_response_text}.
    """
    system_msg = {"role": "system", "content": COUNCIL_SYSTEM}
    user_msg   = {"role": "user",   "content": sanitized_prompt}

    members = {k: v for k, v in COUNCIL_MODELS.items()
               if k in ("member_a", "member_b", "member_c")}

    results = await asyncio.gather(
        *[_query_member(n, c, system_msg, user_msg) for n, c in members.items()]
    )
    return dict(results)

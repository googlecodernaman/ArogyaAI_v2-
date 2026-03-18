"""
Council Orchestrator — 3-stage LLM Council protocol (all-Groq, optimised for speed).

Stage 1: Divergence  — parallel fan-out to Members A, B, C
Stage 2: Convergence — lightweight peer-ranking (compact prompt, fast model)
Stage 3: Synthesis   — Chairman merges top responses into final answer
"""
import asyncio
import json
from .groq_client import query_council_parallel, query_groq, COUNCIL_MODELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    if not text:
        return {}
    try:
        import re
        # Strip Qwen3 chain-of-thought blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"raw": text[:300]}   # cap raw fallback to avoid huge payloads


def _summarise_response(parsed: dict) -> str:
    """
    Compact one-line summary of a council member's response.
    Used in convergence to keep the peer-review prompt small and fast.
    """
    diffs = ", ".join(parsed.get("differentials", [])[:3]) or "none"
    conf  = parsed.get("confidence", "?")
    rf    = parsed.get("red_flag", False)
    return f"Differentials: {diffs} | Confidence: {conf} | RedFlag: {rf}"


def _fallback_synthesis_from_divergence(divergence_results: dict[str, dict], reason: str = "") -> dict:
    """Best-effort synthesis when chairman model is unavailable."""
    differentials: list[str] = []
    next_steps: list[str] = []
    red_flag = False

    for member in divergence_results.values():
        differentials.extend(member.get("differentials", [])[:2])
        next_steps.extend(member.get("next_steps", [])[:2])
        red_flag = red_flag or bool(member.get("red_flag", False))

    # Preserve order while de-duplicating
    dedup_diffs = list(dict.fromkeys(differentials))[:5]
    dedup_steps = list(dict.fromkeys(next_steps))[:5]

    if not dedup_diffs:
        dedup_diffs = ["Insufficient council signal from upstream providers"]
    if not dedup_steps:
        dedup_steps = ["Perform in-person clinical assessment if symptoms persist or worsen"]

    return {
        "final_differentials": dedup_diffs,
        "recommended_next_steps": dedup_steps,
        "confidence": 0.35,
        "red_flag": red_flag,
        "summary": (
            "Council synthesis ran in degraded mode due to upstream LLM unavailability. "
            "Use deterministic triage and local model outputs as primary guidance."
            + (f" ({reason})" if reason else "")
        ),
        "consensus_level": "low",
    }


# ── Stage 1: Divergence ───────────────────────────────────────────────────────

async def run_divergence(sanitized_prompt: str) -> dict[str, dict]:
    """Fan-out to all council members in parallel. Returns parsed JSON per member."""
    raw_responses = await query_council_parallel(sanitized_prompt)
    return {name: _safe_parse_json(text) for name, text in raw_responses.items()}


# ── Stage 2: Convergence ──────────────────────────────────────────────────────

async def run_convergence(
    sanitized_prompt: str,
    divergence_results: dict[str, dict],
) -> dict:
    """
    Lightweight peer review — send compact summaries (not full JSON) to the
    fastest model. Returns ranking + reasoning.
    """
    members = list(divergence_results.keys())
    anon_map = {m: chr(65 + i) for i, m in enumerate(members)}   # A, B, C

    # Build a compact summary block — much smaller than full JSON dumps
    summary_lines = "\n".join(
        f"  {anon_map[m]}: {_summarise_response(divergence_results[m])}"
        for m in members
    )

    review_prompt = (
        f"Case: {sanitized_prompt[:300]}\n\n"
        f"Council member summaries:\n{summary_lines}\n\n"
        f"Task: Rank the responses A, B, C by clinical accuracy and reasoning quality.\n"
        f"Output ONLY this JSON (no other text):\n"
        f'{{"ranking": ["A", "B", "C"], "reasoning": "brief reason"}}'
    )

    # Use the dedicated fast reviewer model for peer ranking.
    # If unavailable, continue with deterministic fallback order.
    try:
        review_text = await query_groq(
            COUNCIL_MODELS["reviewer"]["model"],
            [
                {"role": "system", "content": "You are a clinical peer reviewer. Output only valid JSON."},
                {"role": "user",   "content": review_prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        peer_review = _safe_parse_json(review_text)
        if not peer_review.get("ranking"):
            peer_review = {"ranking": list(anon_map.values()), "reasoning": "default order"}
    except Exception as e:
        peer_review = {
            "ranking": list(anon_map.values()),
            "reasoning": f"fallback order (reviewer unavailable: {type(e).__name__})",
        }

    return {
        "anon_map":           anon_map,
        "peer_review":        peer_review,
        "divergence_results": divergence_results,
    }


# ── Stage 3: Synthesis ────────────────────────────────────────────────────────

async def run_synthesis(sanitized_prompt: str, convergence_data: dict) -> dict:
    """
    Chairman synthesises ALL 3 council responses (Groq + DeepSeek + Mistral)
    into a final clinical answer, weighted by peer-review ranking.
    """
    divergence_results = convergence_data["divergence_results"]
    peer_review        = convergence_data["peer_review"]
    anon_map           = convergence_data["anon_map"]
    ranking            = peer_review.get("ranking", [])

    # Build a full summary of all 3 responses with provider context
    rev_map = {v: k for k, v in anon_map.items()}   # letter → member key
    provider_labels = {
        "member_a": "Groq/Llama-3.3-70B",
        "member_b": "DeepSeek/DeepSeek-Chat",
        "member_c": "Mistral/Mistral-Small",
    }

    all_responses = []
    for letter in ranking if ranking else list(anon_map.values()):
        member_key  = rev_map.get(letter)
        if not member_key:
            continue
        provider = provider_labels.get(member_key, member_key)
        response = divergence_results.get(member_key, {})
        all_responses.append(
            f"[{letter} — {provider}]\n{json.dumps(response, indent=2)}"
        )

    synthesis_prompt = (
        f"Case: {sanitized_prompt[:400]}\n\n"
        f"All 3 independent council responses (ranked best→worst by peer review):\n\n"
        + "\n\n".join(all_responses)
        + f"\n\nPeer ranking reasoning: {peer_review.get('reasoning', '')}\n\n"
        f"Synthesise a final clinical answer that reconciles agreements and flags disagreements. "
        f"Reply ONLY with JSON keys: "
        f"\"final_differentials\" (list), \"recommended_next_steps\" (list), "
        f"\"confidence\" (float 0-1), \"red_flag\" (boolean), \"summary\" (string ≤3 sentences), "
        f"\"consensus_level\" (\"high\"|\"medium\"|\"low\" — based on agreement across providers)."
    )

    try:
        chairman_text = await query_groq(
            COUNCIL_MODELS["chairman"]["model"],
            [
                {"role": "system", "content": "You are the Chairman of a medical AI council. Be concise and accurate."},
                {"role": "user",   "content": synthesis_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        parsed = _safe_parse_json(chairman_text)
        if parsed:
            return parsed
    except Exception as e:
        return _fallback_synthesis_from_divergence(
            divergence_results,
            reason=f"chairman unavailable: {type(e).__name__}",
        )

    return _fallback_synthesis_from_divergence(divergence_results)


# ── Full pipeline (used by non-SSE callers) ───────────────────────────────────

async def orchestrate(sanitized_prompt: str) -> dict:
    divergence  = await run_divergence(sanitized_prompt)
    convergence = await run_convergence(sanitized_prompt, divergence)
    synthesis   = await run_synthesis(sanitized_prompt, convergence)
    return {
        "stage":       "complete",
        "divergence":  divergence,
        "convergence": convergence["peer_review"],
        "synthesis":   synthesis,
    }

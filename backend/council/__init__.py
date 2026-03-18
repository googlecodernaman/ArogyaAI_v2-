"""
MEDORBY Backend — Council Module
Handles LLM Council orchestration, Groq API client, and model configuration.
"""
from .groq_client import query_groq, query_council_parallel, COUNCIL_MODELS, COUNCIL_SYSTEM
from .orchestrator import run_divergence, run_convergence, run_synthesis, orchestrate

__all__ = [
    "query_groq", "query_council_parallel", "COUNCIL_MODELS", "COUNCIL_SYSTEM",
    "run_divergence", "run_convergence", "run_synthesis", "orchestrate",
]

"""
Deterministic backend sanitizer for defense-in-depth.

The client already sanitizes before network transmission. This module ensures
that backend callers and integrations still pass through a deterministic scrubber.
"""

from __future__ import annotations

import re
from typing import Dict

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE), "[DATE]"),
    (re.compile(r"\b(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[ZIP]"),
    (re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sir|Madam)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"), "[NAME]"),
    (re.compile(r"\b(\d{1,3})\s+years?\s+old\b", re.IGNORECASE), r"\1yo"),
)


def sanitize_text(text: str) -> str:
    """Apply deterministic PHI scrubbing rules to free text."""
    sanitized = text or ""
    for pattern, replacement in _PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return " ".join(sanitized.split()).strip()


def sanitize_with_report(text: str) -> Dict[str, object]:
    """Return sanitized text and a minimal audit summary of the transformation."""
    original = text or ""
    sanitized = sanitize_text(original)
    return {
        "sanitized_text": sanitized,
        "changed": sanitized != original,
        "original_length": len(original),
        "sanitized_length": len(sanitized),
    }

"""
Integration tests — Deterministic sanitizer fallback logic.

Tests verify:
 1. Each PII pattern is correctly replaced.
 2. Unknown PII not matched passes through untouched (no false positive).
 3. The server-side fallback in main.py rejects empty-after-sanitation prompts.
 4. Public / clinical text is not mangled.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pytest
from core.sanitizer import sanitize_text, sanitize_with_report


# ── Pattern tests ──────────────────────────────────────────────────────────────

class TestSanitizerPatterns:

    def test_titled_name_is_redacted(self):
        out = sanitize_text("Referring physician Dr. Amit Shah noted normal ECG.")
        assert "[NAME]" in out
        assert "Amit Shah" not in out

    def test_date_slash_format_is_redacted(self):
        out = sanitize_text("DOB listed as 03/15/1978 in records.")
        assert "[DATE]" in out
        assert "03/15/1978" not in out

    def test_date_written_format_is_redacted(self):
        out = sanitize_text("Admitted on January 12, 2024 with chest pain.")
        assert "[DATE]" in out
        assert "January 12" not in out

    def test_phone_number_is_redacted(self):
        out = sanitize_text("Call me at 555-123-4567 after 6pm.")
        assert "[PHONE]" in out
        assert "555-123-4567" not in out

    def test_email_is_redacted(self):
        out = sanitize_text("Send results to patient@example.com for review.")
        assert "[EMAIL]" in out
        assert "patient@example.com" not in out

    def test_ssn_is_redacted(self):
        out = sanitize_text("Insurance card shows SSN 123-45-6789.")
        assert "[SSN]" in out
        assert "123-45-6789" not in out

    def test_multiple_pii_in_one_string(self):
        text = "Dr. Jane Doe, phone 999-888-7777, email dr.doe@clinic.org. DOB 12/01/1985."
        out = sanitize_text(text)
        assert "[NAME]"  in out
        assert "[PHONE]" in out
        assert "[EMAIL]" in out
        assert "[DATE]"  in out
        # Originals gone
        assert "Jane Doe"      not in out
        assert "999-888-7777"  not in out
        assert "dr.doe@clinic" not in out

    def test_age_shorthand_preserved(self):
        """'65 years old' should become '65yo', not be removed."""
        out = sanitize_text("Patient is 65 years old with hypertension.")
        assert "65yo" in out
        assert "years old" not in out

    def test_clean_clinical_text_unchanged(self):
        text = "Crushing chest pain radiating to left arm with diaphoresis and shortness of breath."
        out = sanitize_text(text)
        assert out == text

    def test_whitespace_normalisation(self):
        """Extra whitespace between tokens is collapsed to a single space."""
        out = sanitize_text("Chest   pain   and     fatigue.")
        assert "  " not in out


# ── sanitize_with_report tests ────────────────────────────────────────────────

class TestSanitizeWithReport:

    def test_changed_flag_true_when_pii_found(self):
        r = sanitize_with_report("Patient Dr. Smith, DOB 01/02/2000.")
        assert r["changed"] is True

    def test_changed_flag_false_for_clean_text(self):
        r = sanitize_with_report("Shortness of breath on exertion, ankle oedema.")
        assert r["changed"] is False

    def test_lengths_match_after_sanitisation(self):
        r = sanitize_with_report("Contact jane.doe@hospital.com for labs.")
        assert r["sanitized_length"] == len(r["sanitized_text"])

    def test_empty_input_returns_empty(self):
        r = sanitize_with_report("")
        assert r["sanitized_text"] == ""
        assert r["changed"] is False

    def test_none_treated_as_empty(self):
        """sanitize_text / sanitize_with_report must not break on None."""
        from core.sanitizer import sanitize_text as st
        assert st(None) == ""  # type: ignore[arg-type]


# ── Backend fallback: empty-after-sanitation raises 400 ───────────────────────

class TestBackendSanitizationFallback:
    """
    Verify the _resolve_prompt helper in main.py raises HTTPException(400)
    when the sanitized text is empty.
    """

    def test_empty_prompt_raises_400(self):
        from fastapi import HTTPException
        from main import _resolve_prompt, SymptomRequest

        req = SymptomRequest(sanitized_prompt="")
        with pytest.raises(HTTPException) as exc_info:
            _resolve_prompt(req)
        assert exc_info.value.status_code == 400

    def test_whitespace_only_prompt_raises_400(self):
        from fastapi import HTTPException
        from main import _resolve_prompt, SymptomRequest

        req = SymptomRequest(sanitized_prompt="   ")
        with pytest.raises(HTTPException) as exc_info:
            _resolve_prompt(req)
        assert exc_info.value.status_code == 400

    def test_valid_prompt_passes_through(self):
        from main import _resolve_prompt, SymptomRequest

        req = SymptomRequest(symptom_text="Chest pain and shortness of breath.")
        sanitized, report = _resolve_prompt(req)
        assert len(sanitized) > 0
        assert isinstance(report, dict)

    def test_pii_is_stripped_before_passing(self):
        from main import _resolve_prompt, SymptomRequest

        req = SymptomRequest(
            symptom_text="Dr. Alice Green reports chest tightness. Call 555-000-1234."
        )
        sanitized, report = _resolve_prompt(req)
        assert "Alice Green" not in sanitized
        assert "555-000-1234" not in sanitized
        assert report["changed"] is True

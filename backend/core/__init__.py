"""
MEDORBY Backend — Core Module
Contains configuration and red-flag triage engine.
"""
from .red_flag_engine import evaluate, check_vital_signs, check_symptom_text
from config import get_settings

__all__ = ["evaluate", "check_vital_signs", "check_symptom_text", "get_settings"]

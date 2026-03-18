"""
Feature Extractor — converts a raw vitals dict into the 12-dim normalised
feature vector consumed by the Federated Neural Network.

Expected vitals keys (all optional — missing values use safe defaults):
  age                   int/float  0–120
  systolic_bp           int/float  60–250
  diastolic_bp          int/float  30–150
  heart_rate            int/float  20–220
  spo2                  int/float  50–100
  bmi                   int/float  10–55
  symptom_severity      int        0=none 1=mild 2=moderate 3=severe
  symptom_duration_days int/float  0–365
  has_diabetes          bool/int   0 or 1
  has_hypertension      bool/int   0 or 1
  has_family_history    bool/int   0 or 1
  is_smoker             bool/int   0 or 1
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any

# ── Category metadata (shared with symptom_classifier.py) ─────────────────────

CATEGORY_INFO: dict[str, dict] = {
    "cardiac_emergency": {
        "label":       "Cardiac Emergency",
        "severity":    "critical",
        "description": "Possible acute cardiac event requiring immediate attention",
        "action":      "Seek emergency care immediately. Call 112/911.",
    },
    "cardiac_chronic": {
        "label":       "Chronic Cardiac Condition",
        "severity":    "moderate",
        "description": "Symptoms consistent with chronic heart conditions",
        "action":      "Schedule cardiology consultation. Monitor symptoms.",
    },
    "cardiac_arrhythmia": {
        "label":       "Cardiac Arrhythmia",
        "severity":    "moderate",
        "description": "Symptoms suggesting abnormal heart rhythm",
        "action":      "ECG and Holter monitoring recommended. Cardiology referral.",
    },
    "cardiac_risk": {
        "label":       "Cardiovascular Risk Factors",
        "severity":    "low-moderate",
        "description": "Risk factors for future cardiovascular events",
        "action":      "Lifestyle modifications and risk factor management.",
    },
    "non_cardiac": {
        "label":       "Non-Cardiac",
        "severity":    "low",
        "description": "Symptoms likely not heart-related",
        "action":      "Evaluate for other causes. Primary care follow-up.",
    },
}

# ── Safe defaults (population median values) ──────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "age":                    45,
    "systolic_bp":            120,
    "diastolic_bp":           80,
    "heart_rate":             75,
    "spo2":                   98,
    "bmi":                    25.0,
    "symptom_severity":       1,
    "symptom_duration_days":  1,
    "has_diabetes":           0,
    "has_hypertension":       0,
    "has_family_history":     0,
    "is_smoker":              0,
}

# ── Normalisation ranges ───────────────────────────────────────────────────────

_LOG_DURATION_MAX = math.log1p(365)

_RANGES: dict[str, tuple[float, float]] = {
    "age":         (0.0,  100.0),
    "systolic_bp": (0.0,  200.0),
    "diastolic_bp":(0.0,  130.0),
    "heart_rate":  (0.0,  200.0),
    "spo2":        (0.0,  100.0),
    "bmi":         (0.0,   50.0),
    "symptom_severity": (0.0, 3.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalise(key: str, raw: float) -> float:
    lo, hi = _RANGES[key]
    return _clamp(raw, lo, hi) / hi


def extract_features(vitals: dict) -> np.ndarray:
    """
    Convert a vitals dict → numpy array of shape (12,) in [0, 1].

    Raises ValueError if any value is of an unusable type.
    Missing keys are filled with safe population defaults.
    """
    def _get(key: str) -> float:
        val = vitals.get(key, _DEFAULTS[key])
        if isinstance(val, bool):
            return float(val)
        return float(val)

    feat = np.array([
        _normalise("age",         _get("age")),
        _normalise("systolic_bp", _get("systolic_bp")),
        _normalise("diastolic_bp",_get("diastolic_bp")),
        _normalise("heart_rate",  _get("heart_rate")),
        _normalise("spo2",        _get("spo2")),
        _normalise("bmi",         _get("bmi")),
        _normalise("symptom_severity", _get("symptom_severity")),
        # log-scale duration, then normalise to [0,1]
        math.log1p(_clamp(_get("symptom_duration_days"), 0, 365)) / _LOG_DURATION_MAX,
        float(bool(_get("has_diabetes"))),
        float(bool(_get("has_hypertension"))),
        float(bool(_get("has_family_history"))),
        float(bool(_get("is_smoker"))),
    ], dtype=np.float64)

    return feat


def interpret_vitals(vitals: dict) -> dict:
    """
    Return a human-readable interpretation of each vital sign.
    Useful for displaying alongside the NN prediction.
    """
    results: dict[str, str] = {}

    sbp = vitals.get("systolic_bp")
    dbp = vitals.get("diastolic_bp")
    hr  = vitals.get("heart_rate")
    spo2 = vitals.get("spo2")
    bmi  = vitals.get("bmi")

    if sbp is not None and dbp is not None:
        if sbp > 180 or dbp > 120:
            results["bp_status"] = "hypertensive_crisis"
        elif sbp >= 140 or dbp >= 90:
            results["bp_status"] = "stage2_hypertension"
        elif sbp >= 130 or dbp >= 80:
            results["bp_status"] = "stage1_hypertension"
        elif sbp >= 120:
            results["bp_status"] = "elevated"
        else:
            results["bp_status"] = "normal"

    if hr is not None:
        if hr > 150:
            results["hr_status"] = "severe_tachycardia"
        elif hr > 100:
            results["hr_status"] = "tachycardia"
        elif hr < 40:
            results["hr_status"] = "severe_bradycardia"
        elif hr < 60:
            results["hr_status"] = "bradycardia"
        else:
            results["hr_status"] = "normal"

    if spo2 is not None:
        if spo2 < 85:
            results["spo2_status"] = "critical_hypoxia"
        elif spo2 < 90:
            results["spo2_status"] = "severe_hypoxia"
        elif spo2 < 94:
            results["spo2_status"] = "mild_hypoxia"
        else:
            results["spo2_status"] = "normal"

    if bmi is not None:
        if bmi >= 40:
            results["bmi_status"] = "morbid_obesity"
        elif bmi >= 35:
            results["bmi_status"] = "obesity_class2"
        elif bmi >= 30:
            results["bmi_status"] = "obesity_class1"
        elif bmi >= 25:
            results["bmi_status"] = "overweight"
        elif bmi >= 18.5:
            results["bmi_status"] = "normal"
        else:
            results["bmi_status"] = "underweight"

    return results

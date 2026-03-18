"""
Integration tests — Hybrid decision fusion logic.

Tests verify:
 1. The fusion weights (FNN 45%, text 35%, council 20%) produce the correct winner.
 2. Emergency gate is triggered by red_flag, triage is_emergency, or high-confidence
    cardiac_emergency score.
 3. Probabilities in score_breakdown are normalised (sum = 1).
 4. Fusion correctly falls back when FNN prediction is absent (vitals not supplied).
 5. Deterministic triage short-circuits to emergency without calling LLMs.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pytest
from main import _build_hybrid_decision, _map_synthesis_to_category
from ml.feature_extractor import CATEGORY_INFO

CATEGORIES = list(CATEGORY_INFO.keys())  # 5 classes


# ── Helper factories ───────────────────────────────────────────────────────────

def _triage(is_emergency: bool = False) -> dict:
    return {"is_emergency": is_emergency, "reason": "test", "message": "test"}


def _text_pred(category: str, confidence: float = 0.8) -> dict:
    return {"category": category, "confidence": confidence}


def _nn_pred(category: str, confidence: float = 0.75) -> dict:
    return {"category": category, "confidence": confidence}


def _synthesis(category_keyword: str = "", confidence: float = 0.7, red_flag: bool = False) -> dict:
    """Build a minimal synthesis dict that maps to the desired category via keywords."""
    keyword_map = {
        "cardiac_emergency":  "myocardial infarction stemi",
        "cardiac_arrhythmia": "atrial fibrillation arrhythmia",
        "cardiac_chronic":    "heart failure coronary artery disease",
        "cardiac_risk":       "hypertension cholesterol risk factor",
        "non_cardiac":        "",
    }
    return {
        "summary": keyword_map.get(category_keyword, "general discomfort"),
        "final_differentials": [],
        "confidence": confidence,
        "red_flag": red_flag,
    }


# ── Fusion weight tests ────────────────────────────────────────────────────────

class TestHybridFusionWeights:

    def test_nn_wins_when_all_agree(self):
        """All three sources agree → should return cardiac_risk with high confidence."""
        result = _build_hybrid_decision(
            _triage(), _text_pred("cardiac_risk"), _nn_pred("cardiac_risk"), _synthesis("cardiac_risk")
        )
        assert result["final_category"] == "cardiac_risk"
        assert result["confidence"] > 0.0

    def test_nn_higher_weight_over_text(self):
        """
        NN (45%) says cardiac_emergency; text (35%) says non_cardiac.
        Council is neutralised (confidence=0) to isolate the weight comparison.
        cardiac_emergency score = 0.45*0.9 = 0.405 > non_cardiac = 0.35*0.9 = 0.315.
        """
        result = _build_hybrid_decision(
            _triage(),
            _text_pred("non_cardiac", 0.9),
            _nn_pred("cardiac_emergency", 0.9),
            _synthesis(confidence=0.0),  # council contributes nothing
        )
        assert result["final_category"] == "cardiac_emergency"

    def test_text_wins_when_nn_absent(self):
        """Without vitals (nn_prediction=None), only text + council contribute."""
        result = _build_hybrid_decision(
            _triage(),
            _text_pred("cardiac_arrhythmia", 0.85),
            None,  # no FNN
            _synthesis(),  # council silent
        )
        assert result["final_category"] == "cardiac_arrhythmia"

    def test_score_breakdown_sums_to_one(self):
        result = _build_hybrid_decision(
            _triage(), _text_pred("cardiac_risk"), _nn_pred("non_cardiac"), _synthesis("cardiac_risk")
        )
        total = sum(result["score_breakdown"].values())
        assert abs(total - 1.0) < 1e-6, f"Scores don't sum to 1: {total}"

    def test_score_breakdown_has_all_categories(self):
        result = _build_hybrid_decision(
            _triage(), _text_pred("non_cardiac"), _nn_pred("non_cardiac"), _synthesis()
        )
        for cat in CATEGORIES:
            assert cat in result["score_breakdown"], f"Missing category: {cat}"

    def test_confidence_is_between_zero_and_one(self):
        for cat in CATEGORIES:
            result = _build_hybrid_decision(
                _triage(), _text_pred(cat), _nn_pred(cat), _synthesis(cat)
            )
            assert 0.0 <= result["confidence"] <= 1.0


# ── Emergency gate tests ───────────────────────────────────────────────────────

class TestEmergencyGate:

    def test_triage_emergency_always_escalates(self):
        """If triage says is_emergency, hybrid must also say emergency regardless of ML."""
        result = _build_hybrid_decision(
            _triage(is_emergency=True),
            _text_pred("non_cardiac"),
            _nn_pred("non_cardiac"),
            _synthesis(),
        )
        assert result["is_emergency"] is True
        assert result["severity"] == "critical"
        assert "112" in result["recommended_action"] or "911" in result["recommended_action"]

    def test_synthesis_red_flag_escalates(self):
        """`red_flag=True` in synthesis should trigger emergency gate."""
        result = _build_hybrid_decision(
            _triage(),
            _text_pred("non_cardiac"),
            _nn_pred("non_cardiac"),
            _synthesis(red_flag=True),
        )
        assert result["is_emergency"] is True

    def test_high_confidence_cardiac_emergency_escalates(self):
        """final_category == cardiac_emergency with score > 0.55 triggers gate."""
        result = _build_hybrid_decision(
            _triage(),
            _text_pred("cardiac_emergency", 1.0),
            _nn_pred("cardiac_emergency", 1.0),
            _synthesis("cardiac_emergency", 1.0),
        )
        assert result["is_emergency"] is True

    def test_no_emergency_for_cardiac_risk(self):
        result = _build_hybrid_decision(
            _triage(),
            _text_pred("cardiac_risk"),
            _nn_pred("cardiac_risk"),
            _synthesis("cardiac_risk"),
        )
        assert result["is_emergency"] is False


# ── Sources field tests ────────────────────────────────────────────────────────

class TestSourcesField:

    def test_sources_present_for_all_inputs(self):
        result = _build_hybrid_decision(
            _triage(), _text_pred("cardiac_chronic"), _nn_pred("cardiac_chronic"), _synthesis()
        )
        s = result["sources"]
        assert "text_classifier" in s
        assert "federated_nn" in s
        assert "llm_council" in s

    def test_federated_nn_none_when_no_vitals(self):
        result = _build_hybrid_decision(
            _triage(), _text_pred("non_cardiac"), None, _synthesis()
        )
        assert result["sources"]["federated_nn"]["category"] is None
        assert result["sources"]["federated_nn"]["confidence"] is None

    def test_label_is_human_readable(self):
        result = _build_hybrid_decision(
            _triage(), _text_pred("cardiac_arrhythmia"), None, _synthesis()
        )
        assert result["label"] != "cardiac_arrhythmia"  # should be human label
        assert result["label"] == CATEGORY_INFO["cardiac_arrhythmia"]["label"]


# ── Council category mapping tests ────────────────────────────────────────────

class TestCouncilCategoryMapping:

    @pytest.mark.parametrize("keywords,expected", [
        ("stemi myocardial infarction emergency", "cardiac_emergency"),
        ("atrial fibrillation arrhythmia palpitation", "cardiac_arrhythmia"),
        ("heart failure coronary artery disease angina", "cardiac_chronic"),
        ("hypertension cholesterol risk factor lifestyle", "cardiac_risk"),
        ("muscle strain no cardiac findings", "non_cardiac"),
    ])
    def test_keyword_to_category_mapping(self, keywords: str, expected: str):
        synth = {"summary": keywords, "final_differentials": [], "confidence": 0.7}
        result = _map_synthesis_to_category(synth)
        assert result == expected, f"Got {result!r} for keywords: {keywords!r}"

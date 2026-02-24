"""
Local Symptom Classifier — Lightweight ML model for symptom triage.

Uses a pre-trained TF-IDF + Logistic Regression classifier to predict
the medical category of a symptom description. Runs entirely on-device.

Categories (heart-focused for MVP):
  - cardiac_emergency    → Acute MI, unstable angina, aortic dissection
  - cardiac_chronic      → CAD, heart failure, valvular disease
  - cardiac_arrhythmia   → AFib, SVT, bradycardia
  - cardiac_risk         → Hypertension, high cholesterol, diabetes-related CV risk
  - non_cardiac          → Likely not heart-related (GI, MSK, respiratory, anxiety)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Optional

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "local_model"
MODEL_DIR.mkdir(exist_ok=True)


# ── Training Data (curated symptom-to-category examples) ──────────────────────
# In production this would come from a larger labeled dataset.
# For MVP, we have ~80 hand-crafted examples covering heart-related categories.

TRAINING_DATA = [
    # cardiac_emergency
    ("crushing chest pain radiating to left arm and jaw", "cardiac_emergency"),
    ("severe chest pressure with shortness of breath and sweating", "cardiac_emergency"),
    ("sudden extreme chest tightness with nausea and cold sweat", "cardiac_emergency"),
    ("tearing chest pain radiating to back with unequal blood pressure", "cardiac_emergency"),
    ("chest pain with ST elevation on ECG", "cardiac_emergency"),
    ("acute crushing substernal pain with diaphoresis", "cardiac_emergency"),
    ("sudden cardiac arrest symptoms loss of consciousness no pulse", "cardiac_emergency"),
    ("severe chest pain with troponin elevation and ECG changes", "cardiac_emergency"),
    ("excruciating chest pain and impending feeling of doom", "cardiac_emergency"),
    ("acute onset chest pain with hemodynamic instability", "cardiac_emergency"),
    ("sudden sharp chest pain worse with breathing difficulty", "cardiac_emergency"),
    ("severe retrosternal chest pain with systolic BP drop", "cardiac_emergency"),

    # cardiac_chronic
    ("chronic chest discomfort on exertion relieved by rest", "cardiac_chronic"),
    ("progressive shortness of breath with ankle swelling", "cardiac_chronic"),
    ("fatigue and exercise intolerance with known heart failure", "cardiac_chronic"),
    ("exertional angina stable pattern for months", "cardiac_chronic"),
    ("orthopnea and paroxysmal nocturnal dyspnea", "cardiac_chronic"),
    ("gradually worsening breathlessness with leg edema", "cardiac_chronic"),
    ("known coronary artery disease with stable angina", "cardiac_chronic"),
    ("heart murmur with progressive exercise limitation", "cardiac_chronic"),
    ("dilated cardiomyopathy with reduced ejection fraction", "cardiac_chronic"),
    ("worsening heart failure symptoms NYHA class progression", "cardiac_chronic"),
    ("chronic mitral regurgitation with increasing fatigue", "cardiac_chronic"),
    ("aortic stenosis with syncope on exertion", "cardiac_chronic"),

    # cardiac_arrhythmia
    ("heart palpitations rapid irregular heartbeat", "cardiac_arrhythmia"),
    ("sudden onset rapid heartbeat with dizziness", "cardiac_arrhythmia"),
    ("feeling of skipped heartbeats and racing heart", "cardiac_arrhythmia"),
    ("known atrial fibrillation with irregular pulse", "cardiac_arrhythmia"),
    ("episodes of rapid heart rate with lightheadedness", "cardiac_arrhythmia"),
    ("heart fluttering sensation with fatigue", "cardiac_arrhythmia"),
    ("intermittent palpitations worse with caffeine and stress", "cardiac_arrhythmia"),
    ("slow heart rate below 50 with dizziness and fainting", "cardiac_arrhythmia"),
    ("supraventricular tachycardia episodes", "cardiac_arrhythmia"),
    ("frequent premature ventricular contractions", "cardiac_arrhythmia"),
    ("paroxysmal atrial fibrillation episodes of irregular rhythm", "cardiac_arrhythmia"),
    ("bradycardia with syncope requiring pacemaker evaluation", "cardiac_arrhythmia"),

    # cardiac_risk
    ("high blood pressure readings consistently above 140 90", "cardiac_risk"),
    ("elevated cholesterol LDL above 160 with family history of heart disease", "cardiac_risk"),
    ("diabetes type 2 with uncontrolled blood sugar and chest discomfort", "cardiac_risk"),
    ("obesity BMI over 35 with shortness of breath", "cardiac_risk"),
    ("strong family history of premature coronary artery disease", "cardiac_risk"),
    ("metabolic syndrome with central obesity and high triglycerides", "cardiac_risk"),
    ("smoker for 20 years with occasional chest tightness", "cardiac_risk"),
    ("hypertension stage 2 not responding to single medication", "cardiac_risk"),
    ("prediabetes with elevated hs-CRP and family history CAD", "cardiac_risk"),
    ("sedentary lifestyle with multiple cardiovascular risk factors", "cardiac_risk"),
    ("elevated coronary artery calcium score on CT scan", "cardiac_risk"),
    ("postmenopausal with new onset hypertension and high cholesterol", "cardiac_risk"),

    # non_cardiac
    ("burning stomach pain after eating worse when lying down", "non_cardiac"),
    ("cough with green sputum and fever for 5 days", "non_cardiac"),
    ("sharp pain in chest wall worse when pressing on ribs", "non_cardiac"),
    ("anxiety panic attack with hyperventilation and tingling", "non_cardiac"),
    ("muscle pain in chest after heavy lifting at gym", "non_cardiac"),
    ("acid reflux heartburn relieved by antacids", "non_cardiac"),
    ("headache with nasal congestion and sore throat", "non_cardiac"),
    ("lower back pain radiating to legs", "non_cardiac"),
    ("stomach cramping with diarrhea and nausea", "non_cardiac"),
    ("skin rash with itching on arms and legs", "non_cardiac"),
    ("joint pain and swelling in both knees", "non_cardiac"),
    ("persistent dry cough with mild cold no chest pain", "non_cardiac"),
    ("sore throat with difficulty swallowing and ear pain", "non_cardiac"),
    ("tension headache both sides of head", "non_cardiac"),
    ("insomnia with stress and racing thoughts", "non_cardiac"),
    ("urinary frequency and burning sensation", "non_cardiac"),
]

CATEGORY_INFO = {
    "cardiac_emergency": {
        "label": "Cardiac Emergency",
        "severity": "critical",
        "description": "Possible acute cardiac event requiring immediate attention",
        "action": "Seek emergency care immediately. Call 112/911.",
    },
    "cardiac_chronic": {
        "label": "Chronic Cardiac Condition",
        "severity": "moderate",
        "description": "Symptoms consistent with chronic heart conditions",
        "action": "Schedule cardiology consultation. Monitor symptoms.",
    },
    "cardiac_arrhythmia": {
        "label": "Cardiac Arrhythmia",
        "severity": "moderate",
        "description": "Symptoms suggesting abnormal heart rhythm",
        "action": "ECG and Holter monitoring recommended. Cardiology referral.",
    },
    "cardiac_risk": {
        "label": "Cardiovascular Risk Factors",
        "severity": "low-moderate",
        "description": "Risk factors for future cardiovascular events",
        "action": "Lifestyle modifications and risk factor management.",
    },
    "non_cardiac": {
        "label": "Non-Cardiac",
        "severity": "low",
        "description": "Symptoms likely not heart-related",
        "action": "Evaluate for other causes. Primary care follow-up.",
    },
}


class SymptomClassifier:
    """Local ML classifier for symptom triage."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False

    def train(self):
        """Train the classifier on the curated dataset."""
        texts = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=2048,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

        self.pipeline.fit(texts, labels)
        self.is_trained = True
        print(f"[SymptomClassifier] Trained on {len(texts)} examples, "
              f"{len(set(labels))} categories.")

    def predict(self, symptoms: str) -> dict:
        """
        Predict the medical category and return structured result.
        Returns: {category, label, severity, confidence, probabilities, action}
        """
        if not self.is_trained:
            self.train()

        if not self.pipeline:
            return {"category": "unknown", "confidence": 0}

        # Get prediction and probabilities
        pred = self.pipeline.predict([symptoms])[0]
        probs = self.pipeline.predict_proba([symptoms])[0]
        classes = self.pipeline.classes_

        # Build probability map
        prob_map = {cls: float(prob) for cls, prob in zip(classes, probs)}
        top_confidence = float(max(probs))

        info = CATEGORY_INFO.get(pred, {})

        return {
            "category": pred,
            "label": info.get("label", pred),
            "severity": info.get("severity", "unknown"),
            "confidence": round(top_confidence, 3),
            "description": info.get("description", ""),
            "action": info.get("action", ""),
            "probabilities": {
                CATEGORY_INFO.get(k, {}).get("label", k): round(v, 3)
                for k, v in sorted(prob_map.items(), key=lambda x: -x[1])
            },
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_classifier: Optional[SymptomClassifier] = None


def get_classifier() -> SymptomClassifier:
    """Get singleton classifier instance (trains on first call)."""
    global _classifier
    if _classifier is None:
        _classifier = SymptomClassifier()
        _classifier.train()
    return _classifier

"""
MEDORBY Backend — ML Module
Contains: local symptom classifier (TF-IDF + LogReg),
          federated neural network (numpy, 12→32→64→32→5),
          and structured feature extractor.
"""
from backend.ml.symptom_classifier import get_classifier, SymptomClassifier
from backend.ml.federated_nn import get_federated_nn, HealthPredictionNN, NN_WEIGHT_DIM
from backend.ml.feature_extractor import extract_features, interpret_vitals, CATEGORY_INFO

__all__ = [
    "get_classifier", "SymptomClassifier",
    "get_federated_nn", "HealthPredictionNN", "NN_WEIGHT_DIM",
    "extract_features", "interpret_vitals", "CATEGORY_INFO",
]

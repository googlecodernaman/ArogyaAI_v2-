"""
Federated Neural Network — Lightweight structured health predictor.

Architecture : 12 → 32 → 64 → 32 → 5  (pure numpy, no heavy ML framework)
               ReLU  ReLU  ReLU  Softmax
Total params : 4 773

Input features (12):
  age, systolic_bp, diastolic_bp, heart_rate, spo2, bmi,
  symptom_severity, symptom_duration_days,
  has_diabetes, has_hypertension, has_family_history, is_smoker

Output classes (5):
  cardiac_emergency, cardiac_chronic, cardiac_arrhythmia,
  cardiac_risk, non_cardiac

Federated protocol:
  Client calls  local_train(X, y)  →  compute_weight_delta()
  Server calls  receive_delta(delta)  →  fedavg  →  publish new weights
  Clients call  download_weights(flat)  to apply the global model
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = [
    "cardiac_emergency",
    "cardiac_chronic",
    "cardiac_arrhythmia",
    "cardiac_risk",
    "non_cardiac",
]

INPUT_DIM   = 12
HIDDEN1     = 32
HIDDEN2     = 64
HIDDEN3     = 32
OUTPUT_DIM  = len(CLASSES)

# Flat weight-vector dimension (used by aggregator for DP validation)
NN_WEIGHT_DIM = (
    INPUT_DIM * HIDDEN1 + HIDDEN1      # W1 + b1  = 416
    + HIDDEN1 * HIDDEN2 + HIDDEN2      # W2 + b2  = 2112
    + HIDDEN2 * HIDDEN3 + HIDDEN3      # W3 + b3  = 2080
    + HIDDEN3 * OUTPUT_DIM + OUTPUT_DIM  # W4 + b4  = 165
)  # Total: 4773

WEIGHTS_PATH = Path(__file__).parent / "local_model" / "fnn_weights.json"
WEIGHTS_PATH.parent.mkdir(exist_ok=True)


# ── Activations ───────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy loss; labels are integer class indices."""
    n = len(labels)
    log_p = np.log(probs[np.arange(n), labels] + 1e-9)
    return float(-log_p.mean())


# ── Synthetic pre-training data ───────────────────────────────────────────────

def _generate_synthetic_data(
    n_per_class: int = 120, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic patient records for pre-training.

    Each feature is already normalised to [0, 1] (same as feature_extractor.py):
      0  age               / 100
      1  systolic_bp       / 200
      2  diastolic_bp      / 130
      3  heart_rate        / 200
      4  spo2              / 100
      5  bmi               / 50
      6  symptom_severity  / 3
      7  log(1+duration)   / log(366)
      8  has_diabetes      {0,1}
      9  has_hypertension  {0,1}
      10 has_family_history{0,1}
      11 is_smoker         {0,1}
    """
    rng = np.random.default_rng(seed)

    def _binary(p: float) -> int:
        return int(rng.random() < p)

    rows, lbls = [], []

    # ── 0: cardiac_emergency ──────────────────────────────────────────────────
    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0.40, 0.85),   # age 40–85
            rng.uniform(0.85, 1.00),   # systolic > 170
            rng.uniform(0.77, 1.00),   # diastolic > 100
            rng.uniform(0.65, 1.00),   # HR > 130
            rng.uniform(0.50, 0.90),   # SpO2 50–90 %
            rng.uniform(0.30, 0.80),
            1.00,                      # severe
            rng.uniform(0.00, 0.03),   # acute < 2 days
            _binary(0.40), _binary(0.60), _binary(0.50), _binary(0.50),
        ])
        lbls.append(0)

    # ── 1: cardiac_chronic ────────────────────────────────────────────────────
    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0.45, 0.85),
            rng.uniform(0.65, 0.85),   # systolic 130–170
            rng.uniform(0.55, 0.77),
            rng.uniform(0.35, 0.60),   # HR 70–120
            rng.uniform(0.90, 1.00),   # SpO2 90–100 %
            rng.uniform(0.40, 0.90),
            rng.uniform(0.33, 0.67),   # moderate
            rng.uniform(0.10, 0.60),   # chronic > weeks
            _binary(0.35), _binary(0.55), _binary(0.40), _binary(0.30),
        ])
        lbls.append(1)

    # ── 2: cardiac_arrhythmia ─────────────────────────────────────────────────
    for _ in range(n_per_class):
        hr_val = rng.choice([
            rng.uniform(0.00, 0.30),   # bradycardia < 60
            rng.uniform(0.75, 1.00),   # tachycardia > 150
        ])
        rows.append([
            rng.uniform(0.25, 0.80),
            rng.uniform(0.55, 0.90),
            rng.uniform(0.50, 0.75),
            float(hr_val),
            rng.uniform(0.88, 1.00),
            rng.uniform(0.30, 0.80),
            rng.uniform(0.00, 0.67),
            rng.uniform(0.00, 0.40),
            _binary(0.25), _binary(0.35), _binary(0.30), _binary(0.25),
        ])
        lbls.append(2)

    # ── 3: cardiac_risk ───────────────────────────────────────────────────────
    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0.35, 0.75),
            rng.uniform(0.70, 0.90),   # BP elevated but not crisis
            rng.uniform(0.60, 0.77),
            rng.uniform(0.35, 0.55),
            rng.uniform(0.92, 1.00),
            rng.uniform(0.50, 0.95),
            rng.uniform(0.00, 0.44),   # mild-moderate
            rng.uniform(0.05, 0.80),
            _binary(0.60), _binary(0.70), _binary(0.55), _binary(0.50),
        ])
        lbls.append(3)

    # ── 4: non_cardiac ────────────────────────────────────────────────────────
    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0.15, 0.70),
            rng.uniform(0.55, 0.75),   # normal-ish BP
            rng.uniform(0.46, 0.65),
            rng.uniform(0.35, 0.55),   # normal HR
            rng.uniform(0.95, 1.00),   # good SpO2
            rng.uniform(0.25, 0.80),
            rng.uniform(0.00, 0.67),
            rng.uniform(0.00, 0.60),
            _binary(0.15), _binary(0.20), _binary(0.15), _binary(0.20),
        ])
        lbls.append(4)

    X = np.array(rows, dtype=np.float64)
    y = np.array(lbls, dtype=int)

    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Neural Network ────────────────────────────────────────────────────────────

class HealthPredictionNN:
    """
    Lightweight 3-hidden-layer neural network for structured health prediction.
    Pure numpy — no PyTorch/TensorFlow required.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(0)

        def _glorot(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W1 = _glorot(INPUT_DIM, HIDDEN1)
        self.b1 = np.zeros(HIDDEN1)
        self.W2 = _glorot(HIDDEN1, HIDDEN2)
        self.b2 = np.zeros(HIDDEN2)
        self.W3 = _glorot(HIDDEN2, HIDDEN3)
        self.b3 = np.zeros(HIDDEN3)
        self.W4 = _glorot(HIDDEN3, OUTPUT_DIM)
        self.b4 = np.zeros(OUTPUT_DIM)

        self.is_pretrained = False

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> tuple:
        """Returns (activations, cache) for both predict and backprop."""
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = _relu(z3)
        z4 = a3 @ self.W4 + self.b4
        a4 = _softmax(z4)
        return a4, (X, z1, a1, z2, a2, z3, a3, z4, a4)

    def forward(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self._forward(X)
        return probs

    # ── Backpropagation ───────────────────────────────────────────────────────

    def _backward(
        self,
        cache: tuple,
        labels: np.ndarray,
        l2_lambda: float = 1e-4,
    ) -> dict:
        X, z1, a1, z2, a2, z3, a3, z4, a4 = cache
        n = len(labels)

        # output layer gradient (softmax + cross-entropy combined)
        dz4 = a4.copy()
        dz4[np.arange(n), labels] -= 1.0
        dz4 /= n

        dW4 = a3.T @ dz4 + l2_lambda * self.W4
        db4 = dz4.sum(axis=0)

        da3 = dz4 @ self.W4.T
        dz3 = da3 * _relu_grad(z3)
        dW3 = a2.T @ dz3 + l2_lambda * self.W3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * _relu_grad(z2)
        dW2 = a1.T @ dz2 + l2_lambda * self.W2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(z1)
        dW1 = X.T @ dz1 + l2_lambda * self.W1
        db1 = dz1.sum(axis=0)

        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2,
                    W3=dW3, b3=db3, W4=dW4, b4=db4)

    # ── Training ──────────────────────────────────────────────────────────────

    def local_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
        l2_lambda: float = 1e-4,
    ) -> list[float]:
        """
        Mini-batch SGD on local data.  Called by the client before computing
        the weight delta to send to the federated server.
        Returns per-epoch loss history.
        """
        rng = np.random.default_rng(int(time.time()))
        losses: list[float] = []

        for epoch in range(epochs):
            idx = rng.permutation(len(y))
            X_s, y_s = X[idx], y[idx]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(y_s), batch_size):
                Xb = X_s[start: start + batch_size]
                yb = y_s[start: start + batch_size]

                probs, cache = self._forward(Xb)
                epoch_loss += _cross_entropy_loss(probs, yb)
                n_batches += 1

                grads = self._backward(cache, yb, l2_lambda)
                self.W1 -= lr * grads["W1"]
                self.b1 -= lr * grads["b1"]
                self.W2 -= lr * grads["W2"]
                self.b2 -= lr * grads["b2"]
                self.W3 -= lr * grads["W3"]
                self.b3 -= lr * grads["b3"]
                self.W4 -= lr * grads["W4"]
                self.b4 -= lr * grads["b4"]

            losses.append(epoch_loss / max(n_batches, 1))

        return losses

    def pretrain(self) -> None:
        """Pre-train on synthetic data to bootstrap the global model."""
        X, y = _generate_synthetic_data(n_per_class=120)
        losses = self.local_train(X, y, epochs=40, lr=0.01, batch_size=64)
        acc = self._accuracy(X, y)
        self.is_pretrained = True
        print(
            f"[FederatedNN] Pre-trained on {len(y)} synthetic examples. "
            f"Loss={losses[-1]:.4f}  Acc={acc:.2%}"
        )

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.forward(X).argmax(axis=1)
        return float((preds == y).mean())

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict health category from a single 12-dim feature vector.
        Returns structured dict compatible with the classifier API.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs = self.forward(features.astype(np.float64))[0]
        top_idx = int(probs.argmax())
        top_prob = float(probs[top_idx])

        from ml.feature_extractor import CATEGORY_INFO
        info = CATEGORY_INFO.get(CLASSES[top_idx], {})

        return {
            "category":     CLASSES[top_idx],
            "label":        info.get("label", CLASSES[top_idx]),
            "severity":     info.get("severity", "unknown"),
            "confidence":   round(top_prob, 3),
            "description":  info.get("description", ""),
            "action":       info.get("action", ""),
            "probabilities": {
                CLASSES[i]: round(float(p), 3)
                for i, p in enumerate(probs)
            },
            "model":        "federated_nn",
        }

    # ── Weight serialisation ──────────────────────────────────────────────────

    def get_weights_flat(self) -> list[float]:
        """Flatten all parameters into a single list (length = NN_WEIGHT_DIM)."""
        parts = [self.W1, self.b1, self.W2, self.b2,
                 self.W3, self.b3, self.W4, self.b4]
        return np.concatenate([p.ravel() for p in parts]).tolist()

    def set_weights_flat(self, flat: list[float]) -> None:
        """Restore all parameters from a flat list."""
        arr = np.array(flat, dtype=np.float64)
        cursor = 0

        def _slice(shape):
            nonlocal cursor
            size = int(np.prod(shape))
            chunk = arr[cursor: cursor + size].reshape(shape)
            cursor += size
            return chunk

        self.W1 = _slice((INPUT_DIM, HIDDEN1))
        self.b1 = _slice((HIDDEN1,))
        self.W2 = _slice((HIDDEN1, HIDDEN2))
        self.b2 = _slice((HIDDEN2,))
        self.W3 = _slice((HIDDEN2, HIDDEN3))
        self.b3 = _slice((HIDDEN3,))
        self.W4 = _slice((HIDDEN3, OUTPUT_DIM))
        self.b4 = _slice((OUTPUT_DIM,))

    def compute_weight_delta(self, old_weights: list[float]) -> list[float]:
        """Return (current_weights − old_weights) as the federated gradient delta."""
        old = np.array(old_weights, dtype=np.float64)
        new = np.array(self.get_weights_flat(), dtype=np.float64)
        return (new - old).tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        data = {
            "weights": self.get_weights_flat(),
            "is_pretrained": self.is_pretrained,
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path = WEIGHTS_PATH) -> bool:
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.set_weights_flat(data["weights"])
            self.is_pretrained = data.get("is_pretrained", False)
            return True
        except Exception as e:
            print(f"[FederatedNN] Could not load weights: {e}")
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────

_nn: Optional[HealthPredictionNN] = None


def get_federated_nn() -> HealthPredictionNN:
    """Return singleton NN (pre-trained on first call)."""
    global _nn
    if _nn is None:
        _nn = HealthPredictionNN()
        if not _nn.load():
            _nn.pretrain()
            _nn.save()
    return _nn

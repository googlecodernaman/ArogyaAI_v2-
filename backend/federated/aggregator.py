"""
Federated Aggregator — implements FedAvg (Federated Averaging).
Collects DP-noised adapter deltas from clients, aggregates them,
and maintains a versioned Global Model Store.
"""
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np
from .dp_privacy import apply_dp, validate_update
from backend.ml.federated_nn import NN_WEIGHT_DIM, get_federated_nn

ADAPTER_STORE_PATH = Path(__file__).parent.parent / "global_adapters"
ADAPTER_STORE_PATH.mkdir(exist_ok=True)

# In-memory buffer of pending client updates
_pending_updates: list[dict] = []
_current_version: int = 0
_global_weights: Optional[list[float]] = None

ADAPTER_DIM = NN_WEIGHT_DIM


def _ensure_global_weights() -> list[float]:
    """Load the global model baseline from the local federated NN singleton."""
    global _global_weights
    if _global_weights is None:
        nn = get_federated_nn()
        _global_weights = nn.get_weights_flat()
    return _global_weights


def receive_update(client_id: str, raw_gradients: list[float]) -> dict:
    """
    Accept a raw gradient update from a client device.
    Applies DP (clip + noise) server-side as a second layer of protection,
    then buffers the update.
    """
    if not validate_update(raw_gradients, ADAPTER_DIM):
        return {"status": "error", "message": f"Expected {ADAPTER_DIM}-dim update."}

    # Apply server-side DP as defense-in-depth
    dp_update = apply_dp(raw_gradients, clip_norm=1.0, noise_multiplier=0.8)

    _pending_updates.append({
        "client_id": client_id,
        "update": dp_update,
        "timestamp": time.time(),
    })

    return {
        "status": "accepted",
        "pending_count": len(_pending_updates),
        "message": "Update received and DP-processed.",
    }


def aggregate(min_clients: int = 2) -> Optional[dict]:
    """
    FedAvg: average all pending updates once min_clients threshold is met.
    Saves the new global adapter and increments version.
    Returns the new adapter metadata or None if threshold not met.
    """
    global _current_version

    if len(_pending_updates) < min_clients:
        return None

    current = np.array(_ensure_global_weights(), dtype=np.float64)

    # FedAvg: average the submitted noised deltas
    all_updates = np.array([u["update"] for u in _pending_updates])
    avg_delta = np.mean(all_updates, axis=0)
    new_global_weights = (current + avg_delta).tolist()

    _current_version += 1
    version_path = ADAPTER_STORE_PATH / f"adapter_v{_current_version}.json"
    metadata = {
        "version": _current_version,
        "num_clients": len(_pending_updates),
        "timestamp": time.time(),
        "adapter": new_global_weights,
        "avg_delta": avg_delta.tolist(),
        "adapter_dim": ADAPTER_DIM,
    }

    with open(version_path, "w") as f:
        json.dump(metadata, f)

    # Promote this aggregated model as the active global model and persist to local NN.
    global _global_weights
    _global_weights = new_global_weights
    nn = get_federated_nn()
    nn.set_weights_flat(new_global_weights)
    nn.save()

    _pending_updates.clear()

    return {
        "status": "aggregated",
        "version": _current_version,
        "num_clients": metadata["num_clients"],
        "adapter_path": str(version_path),
    }


def get_latest_adapter() -> Optional[dict]:
    """Return the latest global adapter weights for client download."""
    if _current_version == 0:
        return None
    version_path = ADAPTER_STORE_PATH / f"adapter_v{_current_version}.json"
    if version_path.exists():
        with open(version_path) as f:
            return json.load(f)
    return None


def get_status() -> dict:
    """Return current aggregator status."""
    return {
        "current_version": _current_version,
        "pending_updates": len(_pending_updates),
        "expected_update_dim": ADAPTER_DIM,
        "adapter_store": str(ADAPTER_STORE_PATH),
    }

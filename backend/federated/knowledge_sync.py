"""
Federated Knowledge Sync

Accepts sanitized local knowledge snippets from edge nodes and stores them in a
shared JSON file under the knowledge base so they are picked up by the RAG index.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
FEDERATED_KB_PATH = KB_DIR / "federated_updates.json"

KB_DIR.mkdir(exist_ok=True)


def _read_entries() -> list[dict[str, Any]]:
    if not FEDERATED_KB_PATH.exists():
        return []
    try:
        with open(FEDERATED_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _write_entries(entries: list[dict[str, Any]]) -> None:
    with open(FEDERATED_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=True, indent=2)


def receive_knowledge_update(
    client_id: str,
    topic: str,
    content: str,
    source: str = "federated_node",
) -> dict[str, Any]:
    """Append a sanitized federated knowledge entry and return summary metadata."""
    entries = _read_entries()
    entry = {
        "id": f"fkb_{uuid.uuid4().hex[:12]}",
        "topic": topic[:120].strip() or "Federated Case Insight",
        "source": source[:120].strip() or "federated_node",
        "content": content[:2500].strip(),
        "node_id_hash": client_id[:32],
        "timestamp": time.time(),
    }
    entries.append(entry)
    _write_entries(entries)

    return {
        "status": "accepted",
        "entry_id": entry["id"],
        "total_entries": len(entries),
        "path": str(FEDERATED_KB_PATH),
    }


def get_federated_kb_status() -> dict[str, Any]:
    entries = _read_entries()
    return {
        "entries": len(entries),
        "path": str(FEDERATED_KB_PATH),
        "latest_topic": entries[-1]["topic"] if entries else None,
    }

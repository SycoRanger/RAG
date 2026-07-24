"""
backend/utils/logger.py
-------------------------
Lightweight CSV logging of every query sent through POST /api/query.
Dependency-free (no logging frameworks) so the file is easy to inspect
directly, e.g. for the assignment's "logging user queries" requirement.
"""

import csv
import os
from datetime import datetime, timezone

from backend.config import settings

_FIELDNAMES = [
    "timestamp_utc",
    "namespace",
    "question",
    "num_chunks_used",
    "used_context",
    "confidence",
    "answer_preview",
]


def _ensure_log_file():
    os.makedirs(settings.log_dir, exist_ok=True)
    if not os.path.exists(settings.query_log_path):
        with open(settings.query_log_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()


def log_query(
    namespace: str,
    question: str,
    num_chunks_used: int,
    used_context: bool,
    confidence: float,
    answer: str,
) -> None:
    _ensure_log_file()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "namespace": namespace,
        "question": question,
        "num_chunks_used": num_chunks_used,
        "used_context": used_context,
        "confidence": confidence,
        "answer_preview": (answer[:120] + "…") if len(answer) > 120 else answer,
    }
    with open(settings.query_log_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(row)


def read_recent_logs(limit: int = 50) -> list[dict]:
    if not os.path.exists(settings.query_log_path):
        return []
    with open(settings.query_log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:][::-1]

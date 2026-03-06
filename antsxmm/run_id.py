from __future__ import annotations

import re


def normalize_run_id(run: str | int | None) -> str:
    """Normalize run identifiers to canonical BIDS form (run-XX)."""
    if run is None:
        return "run-01"
    s = str(run).strip()
    if not s or s.lower() in {"nan", "none"}:
        return "run-01"
    m = re.search(r"run-(\d+)", s)
    if m:
        return f"run-{int(m.group(1)):02d}"
    m = re.search(r"(?:^|_)(?:r)(\d+)(?:[_.-]|$)", s)
    if m:
        return f"run-{int(m.group(1)):02d}"
    m = re.search(r"(\d+)", s)
    if m:
        return f"run-{int(m.group(1)):02d}"
    return "run-01"

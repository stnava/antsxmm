from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import DiscoveredRun, RunKey, SessionKey


@dataclass(frozen=True)
class CsvValidation:
    exists: bool
    is_valid: bool | None
    issue: str | None


@dataclass(frozen=True)
class OutputInventory:
    discovered_runs: tuple[DiscoveredRun, ...]
    status_by_session: dict[SessionKey, dict | None]


def scan_output_inventory(output_dir: str | Path, project_id: str) -> OutputInventory:
    output_dir = Path(output_dir)
    project_root = output_dir / project_id
    runs: list[DiscoveredRun] = []
    status_by_session: dict[SessionKey, dict | None] = {}
    if not project_root.exists():
        return OutputInventory(discovered_runs=(), status_by_session={})

    for subject_dir in sorted(project_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if not session_dir.is_dir():
                continue
            session = SessionKey(project_id=project_id, subject_id=subject_dir.name, session_id=session_dir.name)
            status_by_session[session] = _load_status_file(session_dir / ".antsxmm_status.json")
            for modality_dir in sorted(session_dir.iterdir()):
                if not modality_dir.is_dir():
                    continue
                for run_dir in sorted(modality_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    mmwide_candidates = sorted(run_dir.glob("*+mmwide.csv"))
                    runs.append(
                        DiscoveredRun(
                            session=session,
                            run=RunKey(modality=modality_dir.name, run_id=run_dir.name),
                            output_dir=run_dir,
                            mmwide_csv=mmwide_candidates[0] if mmwide_candidates else None,
                        )
                    )
    return OutputInventory(discovered_runs=tuple(runs), status_by_session=status_by_session)


def _load_status_file(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_invalid": True, "path": str(path)}


def validate_mmwide_csv(path: Path) -> CsvValidation:
    if not path.exists():
        return CsvValidation(exists=False, is_valid=None, issue="missing")
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return CsvValidation(exists=True, is_valid=False, issue="empty_header")
            row = next(reader, None)
            if row is None:
                return CsvValidation(exists=True, is_valid=False, issue="no_rows")
            if not any(str(col).strip() for col in header):
                return CsvValidation(exists=True, is_valid=False, issue="blank_header")
    except Exception as exc:
        return CsvValidation(exists=True, is_valid=False, issue=f"parse_error:{exc.__class__.__name__}")
    return CsvValidation(exists=True, is_valid=True, issue=None)

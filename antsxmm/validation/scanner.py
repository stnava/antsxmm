from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import DiscoveredRun, RunKey, SessionKey
from .schema_profiles import get_schema_profile


@dataclass(frozen=True)
class CsvValidation:
    exists: bool
    is_valid: bool | None
    issue: str | None
    columns: tuple[str, ...] = ()
    row_count: int = 0
    profile_name: str | None = None
    strict_schema_applied: bool = False
    metric_matches: tuple[str, ...] = ()


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


def validate_mmwide_csv(path: Path, *, modality: str | None = None, strict_schema: bool = False) -> CsvValidation:
    profile = get_schema_profile(modality or "default")
    if not path.exists():
        return CsvValidation(exists=False, is_valid=None, issue="missing", profile_name=profile.modality, strict_schema_applied=strict_schema)
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return CsvValidation(exists=True, is_valid=False, issue="empty_header", profile_name=profile.modality, strict_schema_applied=strict_schema)
            normalized_header = tuple(str(col).strip() for col in header)
            if not any(normalized_header):
                return CsvValidation(exists=True, is_valid=False, issue="blank_header", columns=normalized_header, profile_name=profile.modality, strict_schema_applied=strict_schema)
            lowered_nonblank = [col.lower() for col in normalized_header if col]
            if len(lowered_nonblank) != len(set(lowered_nonblank)):
                return CsvValidation(exists=True, is_valid=False, issue="duplicate_columns", columns=normalized_header, profile_name=profile.modality, strict_schema_applied=strict_schema)
            rows = list(reader)
            if not rows:
                return CsvValidation(exists=True, is_valid=False, issue="no_rows", columns=normalized_header, profile_name=profile.modality, strict_schema_applied=strict_schema)
            if len(normalized_header) < profile.min_columns:
                return CsvValidation(exists=True, is_valid=False, issue="schema_too_few_columns", columns=normalized_header, row_count=len(rows), profile_name=profile.modality, strict_schema_applied=strict_schema)
            accepted_identifiers = set(profile.normalized_identifier_columns())
            if accepted_identifiers and not any(col.lower() in accepted_identifiers for col in normalized_header if col):
                return CsvValidation(exists=True, is_valid=False, issue=f"schema_missing_identifier:{profile.modality}", columns=normalized_header, row_count=len(rows), profile_name=profile.modality, strict_schema_applied=strict_schema)
            metric_matches = profile.match_metric_columns(normalized_header)
            if strict_schema and profile.strict_enabled() and len(metric_matches) < profile.strict_min_metric_matches:
                return CsvValidation(
                    exists=True,
                    is_valid=False,
                    issue=f"strict_schema_missing_metrics:{profile.modality}",
                    columns=normalized_header,
                    row_count=len(rows),
                    profile_name=profile.modality,
                    strict_schema_applied=True,
                    metric_matches=metric_matches,
                )
    except Exception as exc:
        return CsvValidation(exists=True, is_valid=False, issue=f"parse_error:{exc.__class__.__name__}", profile_name=profile.modality, strict_schema_applied=strict_schema)
    return CsvValidation(
        exists=True,
        is_valid=True,
        issue=None,
        columns=normalized_header,
        row_count=len(rows),
        profile_name=profile.modality,
        strict_schema_applied=strict_schema,
        metric_matches=metric_matches,
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PATTERN = "*mmwidemerged.csv"
DEFAULT_PREFER = "processed-first"
PREFER_CHOICES = {"processed-first", "pymm-first", "newest", "largest", "error"}


@dataclass(frozen=True)
class FileRecord:
    source_path: Path
    relative_path: str
    fingerprint: str
    project_id: str | None
    subject_id: str | None
    session_id: str | None
    modality: str | None
    run_id: str | None
    source_root: str | None
    entity_id: str | None
    size: int
    mtime_ns: int

    def to_state(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "fingerprint": self.fingerprint,
            "project_id": self.project_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "modality": self.modality,
            "run_id": self.run_id,
            "source_root": self.source_root,
            "entity_id": self.entity_id,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class AggregateResult:
    output_path: Path
    state_path: Path
    rejects_path: Path
    scanned: int
    read: int
    rows_written: int
    rejected: int
    reused_existing: bool
    incremental: bool


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None



def _record_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    size = int(stat.st_size)
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    return size, mtime_ns, f"{size}:{mtime_ns}"



def _parse_from_path(root: Path, path: Path) -> dict[str, str | None]:
    rel = path.relative_to(root)
    parts = list(rel.parts)

    subject_idx = next((i for i, part in enumerate(parts) if part.startswith("sub-")), None)
    subject_id = parts[subject_idx] if subject_idx is not None else None
    session_id = parts[subject_idx + 1] if subject_idx is not None and subject_idx + 1 < len(parts) and parts[subject_idx + 1].startswith("ses-") else None
    modality = parts[subject_idx + 2] if subject_idx is not None and subject_idx + 2 < len(parts) else None
    run_id = parts[subject_idx + 3] if subject_idx is not None and subject_idx + 3 < len(parts) and parts[subject_idx + 3].startswith("run-") else None
    project_id = parts[subject_idx - 1] if subject_idx is not None and subject_idx >= 1 else None
    source_root = parts[subject_idx - 2] if subject_idx is not None and subject_idx >= 2 else None

    return {
        "project_id": project_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "modality": modality,
        "run_id": run_id,
        "source_root": source_root,
    }



def _parse_from_filename(path: Path) -> dict[str, str | None]:
    stem = path.name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    tokens = stem.split("+")
    if len(tokens) < 6:
        return {
            "project_id": None,
            "subject_id": None,
            "session_id": None,
            "modality": None,
            "run_id": None,
        }
    return {
        "project_id": _safe_text(tokens[0]),
        "subject_id": _safe_text(tokens[1]),
        "session_id": _safe_text(tokens[2]),
        "modality": _safe_text(tokens[3]),
        "run_id": _safe_text(tokens[4]),
    }



def _resolve_identity(root: Path, path: Path) -> tuple[FileRecord | None, str | None]:
    path_bits = _parse_from_path(root, path)
    name_bits = _parse_from_filename(path)

    issues: list[str] = []
    resolved: dict[str, str | None] = {}
    for key in ("project_id", "subject_id", "session_id", "modality", "run_id"):
        path_value = path_bits.get(key)
        name_value = name_bits.get(key)
        if path_value and name_value and path_value != name_value:
            issues.append(f"{key} mismatch path={path_value} filename={name_value}")
        resolved[key] = path_value or name_value

    resolved["source_root"] = path_bits.get("source_root")

    missing = [key for key in ("project_id", "subject_id", "session_id", "modality", "run_id") if not resolved.get(key)]
    if missing:
        issues.append("missing " + ",".join(missing))

    if issues:
        return None, "; ".join(issues)

    size, mtime_ns, fingerprint = _record_fingerprint(path)
    entity_id = "|".join(
        [
            resolved["project_id"],
            resolved["subject_id"],
            resolved["session_id"],
            resolved["modality"],
            resolved["run_id"],
        ]
    )
    return FileRecord(
        source_path=path,
        relative_path=str(path.relative_to(root)),
        fingerprint=fingerprint,
        project_id=resolved["project_id"],
        subject_id=resolved["subject_id"],
        session_id=resolved["session_id"],
        modality=resolved["modality"],
        run_id=resolved["run_id"],
        source_root=resolved["source_root"],
        entity_id=entity_id,
        size=size,
        mtime_ns=mtime_ns,
    ), None



def discover_merged_csvs(root: str | Path, pattern: str = DEFAULT_PATTERN) -> tuple[list[FileRecord], pd.DataFrame]:
    root_path = Path(root).expanduser().resolve()
    files = sorted(root_path.rglob(pattern))
    records: list[FileRecord] = []
    rejects: list[dict[str, Any]] = []

    for path in files:
        record, reason = _resolve_identity(root_path, path)
        if record is None:
            rejects.append({"source_path": str(path), "reason": reason or "identity_resolution_failed"})
            continue
        records.append(record)

    rejects_df = pd.DataFrame(rejects) if rejects else pd.DataFrame(columns=["source_path", "reason"])
    return records, rejects_df



def _load_one_csv(record: FileRecord) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    try:
        df = pd.read_csv(record.source_path)
    except Exception as exc:
        return None, {"source_path": str(record.source_path), "reason": f"read_failed: {exc}"}

    if df.empty:
        return None, {"source_path": str(record.source_path), "reason": "empty_csv"}

    if len(df) > 1:
        df = df.iloc[[-1]].copy()
    else:
        df = df.copy()

    for col in list(df.columns):
        if col in {"project_id", "subject_id", "session_id", "modality", "run_id", "entity_id", "source_root", "source_path", "source_mtime_ns", "source_size", "source_fingerprint"}:
            df = df.rename(columns={col: f"input_{col}"})

    df.insert(0, "source_fingerprint", record.fingerprint)
    df.insert(0, "source_size", record.size)
    df.insert(0, "source_mtime_ns", record.mtime_ns)
    df.insert(0, "source_path", str(record.source_path))
    df.insert(0, "source_root", record.source_root)
    df.insert(0, "entity_id", record.entity_id)
    df.insert(0, "run_id", record.run_id)
    df.insert(0, "modality", record.modality)
    df.insert(0, "session_id", record.session_id)
    df.insert(0, "subject_id", record.subject_id)
    df.insert(0, "project_id", record.project_id)
    return df, None



def _prefer_rank(source_root: str | None, prefer: str) -> int:
    root = (source_root or "").lower()
    if prefer == "processed-first":
        return 0 if root == "processed" else 1
    if prefer == "pymm-first":
        return 0 if root == "pymm" else 1
    return 0



def _dedupe(df: pd.DataFrame, prefer: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    dedupe_cols = ["project_id", "subject_id", "session_id", "modality", "run_id"]
    work = df.copy()
    work["_prefer_rank"] = work["source_root"].map(lambda x: _prefer_rank(x, prefer))
    work["_source_path_rank"] = work["source_path"].astype(str)

    if prefer == "error":
        dupes = work[work.duplicated(subset=dedupe_cols, keep=False)]
        if not dupes.empty:
            sample = dupes[dedupe_cols + ["source_path"]].sort_values(dedupe_cols + ["source_path"])
            raise ValueError("duplicate entity rows detected under --prefer=error\n" + sample.to_string(index=False))
        return work.drop(columns=["_prefer_rank", "_source_path_rank"])

    ascending = [True, True, False, True]
    sort_cols = ["_prefer_rank", "source_mtime_ns", "source_size", "_source_path_rank"]
    if prefer == "newest":
        sort_cols = ["source_mtime_ns", "source_size", "_prefer_rank", "_source_path_rank"]
        ascending = [False, False, True, True]
    elif prefer == "largest":
        sort_cols = ["source_size", "source_mtime_ns", "_prefer_rank", "_source_path_rank"]
        ascending = [False, False, True, True]

    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    work = work.drop_duplicates(subset=dedupe_cols, keep="first")
    work = work.drop(columns=["_prefer_rank", "_source_path_rank"])
    return work.sort_values(dedupe_cols, kind="mergesort").reset_index(drop=True)



def _read_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)



def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)



def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text(encoding="utf-8"))



def _write_state(path: Path, root: Path, records: list[FileRecord], pattern: str, prefer: str) -> None:
    payload = {
        "root": str(root),
        "pattern": pattern,
        "prefer": prefer,
        "files": {record.relative_path: record.to_state() for record in records},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")



def aggregate_merged_tables(
    root: str | Path,
    output: str | Path,
    pattern: str = DEFAULT_PATTERN,
    state_path: str | Path | None = None,
    rejects_path: str | Path | None = None,
    incremental: bool = True,
    prefer: str = DEFAULT_PREFER,
) -> AggregateResult:
    if prefer not in PREFER_CHOICES:
        raise ValueError(f"Unsupported prefer policy: {prefer}")

    root_path = Path(root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    state_path = Path(state_path).expanduser().resolve() if state_path else output_path.with_suffix(output_path.suffix + ".state.json")
    rejects_path = Path(rejects_path).expanduser().resolve() if rejects_path else output_path.with_suffix(output_path.suffix + ".rejects.csv")

    records, rejects_df = discover_merged_csvs(root_path, pattern=pattern)
    current_by_rel = {record.relative_path: record for record in records}

    old_state = _read_state(state_path) if incremental else {"files": {}}
    old_files = old_state.get("files", {}) or {}

    full_rebuild = not incremental or not output_path.exists() or not state_path.exists()
    reused_existing = False

    if full_rebuild:
        read_records = list(records)
        existing_df = pd.DataFrame()
    else:
        changed_or_new = {
            rel for rel, rec in current_by_rel.items()
            if rel not in old_files or old_files[rel].get("fingerprint") != rec.fingerprint
        }
        deleted = {rel for rel in old_files.keys() if rel not in current_by_rel}
        affected_entity_ids = {
            current_by_rel[rel].entity_id for rel in changed_or_new if current_by_rel[rel].entity_id
        }
        affected_entity_ids.update(
            old_files[rel].get("entity_id") for rel in deleted if old_files[rel].get("entity_id")
        )

        if not affected_entity_ids:
            existing_df = _read_existing_table(output_path)
            read_records = []
            reused_existing = True
        else:
            existing_df = _read_existing_table(output_path)
            if not existing_df.empty and "entity_id" in existing_df.columns:
                existing_df = existing_df[~existing_df["entity_id"].isin(sorted(affected_entity_ids))].copy()
            read_records = [record for record in records if record.entity_id in affected_entity_ids]

    loaded_frames: list[pd.DataFrame] = []
    reject_rows = rejects_df.to_dict(orient="records") if not rejects_df.empty else []
    for record in read_records:
        df, reject = _load_one_csv(record)
        if reject is not None:
            reject_rows.append(reject)
            continue
        loaded_frames.append(df)

    rebuilt_df = pd.concat(loaded_frames, ignore_index=True, sort=False) if loaded_frames else pd.DataFrame()
    combined_df = pd.concat([existing_df, rebuilt_df], ignore_index=True, sort=False) if not existing_df.empty or not rebuilt_df.empty else pd.DataFrame()
    final_df = _dedupe(combined_df, prefer=prefer)

    _write_table(final_df, output_path)
    final_rejects_df = pd.DataFrame(reject_rows) if reject_rows else pd.DataFrame(columns=["source_path", "reason"])
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    final_rejects_df.to_csv(rejects_path, index=False)
    _write_state(state_path, root_path, records, pattern=pattern, prefer=prefer)

    return AggregateResult(
        output_path=output_path,
        state_path=state_path,
        rejects_path=rejects_path,
        scanned=len(records) + len(reject_rows),
        read=len(read_records),
        rows_written=int(final_df.shape[0]),
        rejected=int(final_rejects_df.shape[0]),
        reused_existing=reused_existing,
        incremental=incremental,
    )

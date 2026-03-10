from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingCode(StrEnum):
    MISSING_RUN_DIR = "missing_run_dir"
    UNEXPECTED_RUN_DIR = "unexpected_run_dir"
    MISSING_MMWIDE_CSV = "missing_mmwide_csv"
    INVALID_MMWIDE_CSV = "invalid_mmwide_csv"
    EMPTY_MMWIDE_CSV = "empty_mmwide_csv"
    MISSING_STATUS_FILE = "missing_status_file"
    INVALID_STATUS_FILE = "invalid_status_file"
    STATUS_SESSION_MISMATCH = "status_session_mismatch"
    ORPHAN_OUTPUT = "orphan_output"


@dataclass(frozen=True, order=True)
class SessionKey:
    project_id: str
    subject_id: str
    session_id: str

    @property
    def label(self) -> str:
        return f"{self.subject_id}/{self.session_id}"


@dataclass(frozen=True, order=True)
class RunKey:
    modality: str
    run_id: str


@dataclass(frozen=True)
class ExpectedArtifact:
    session: SessionKey
    run: RunKey
    output_dir: Path
    mmwide_csv: Path
    status_file: Path


@dataclass(frozen=True)
class DiscoveredRun:
    session: SessionKey
    run: RunKey
    output_dir: Path
    mmwide_csv: Path | None


@dataclass(frozen=True)
class ValidationFinding:
    code: FindingCode
    severity: Severity
    session: SessionKey
    run: RunKey | None
    path: Path | None
    message: str


@dataclass(frozen=True)
class RunValidationRecord:
    expected: ExpectedArtifact
    dir_exists: bool
    mmwide_exists: bool
    mmwide_valid: bool | None
    findings: tuple[ValidationFinding, ...] = ()
    csv_columns: tuple[str, ...] = ()
    csv_row_count: int = 0
    csv_issue: str | None = None
    csv_profile: str | None = None
    strict_schema_applied: bool = False
    csv_metric_matches: tuple[str, ...] = ()

    @property
    def session(self) -> SessionKey:
        return self.expected.session

    @property
    def run(self) -> RunKey:
        return self.expected.run


@dataclass(frozen=True)
class StudyValidationReport:
    records: tuple[RunValidationRecord, ...]
    findings: tuple[ValidationFinding, ...]
    expected_sessions: tuple[SessionKey, ...]
    discovered_sessions: tuple[SessionKey, ...]
    status_by_session: Mapping[SessionKey, dict | None] = field(default_factory=dict)
    strict_schema: bool = False

    def records_for_session(self, session: SessionKey) -> tuple[RunValidationRecord, ...]:
        return tuple(record for record in self.records if record.session == session)

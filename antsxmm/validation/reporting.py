from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import FindingCode, StudyValidationReport


@dataclass(frozen=True)
class IssueCodeRow:
    code: str
    count: int


def build_issue_code_table(report: StudyValidationReport) -> list[IssueCodeRow]:
    counts = Counter(str(f.code) for f in report.findings)
    return [IssueCodeRow(code=code, count=counts[code]) for code in sorted(counts)]


def build_session_issue_counts(report: StudyValidationReport) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for finding in report.findings:
        counts[finding.session.label] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

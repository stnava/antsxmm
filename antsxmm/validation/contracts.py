from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import ExpectedArtifact, RunKey, SessionKey
from ..tree import predict_tree


def iter_expected_artifacts(
    bids_project_dir: str | Path,
    output_dir: str | Path,
    *,
    participant_labels: Iterable[str] | None = None,
) -> tuple[ExpectedArtifact, ...]:
    bids_project_dir = Path(bids_project_dir)
    output_dir = Path(output_dir)
    project = bids_project_dir.name
    wanted = None if participant_labels is None else {str(x).strip() for x in participant_labels if str(x).strip()}
    artifacts: list[ExpectedArtifact] = []

    for subject_dir in sorted(bids_project_dir.glob("sub-*")):
        subject = subject_dir.name
        if wanted and subject not in wanted:
            continue
        _, _, tree = predict_tree(subject_dir)
        for session_id, runs in sorted(tree.items()):
            session = SessionKey(project_id=project, subject_id=subject, session_id=session_id)
            status_file = output_dir / project / subject / session_id / ".antsxmm_status.json"
            for modality, run_id in sorted(runs, key=lambda item: (item[0], item[1])):
                run = RunKey(modality=modality, run_id=run_id)
                run_dir = output_dir / project / subject / session_id / modality / run_id
                mmwide_name = f"{project}+{subject}+{session_id}+{modality}+{run_id}+mmwide.csv"
                artifacts.append(
                    ExpectedArtifact(
                        session=session,
                        run=run,
                        output_dir=run_dir,
                        mmwide_csv=run_dir / mmwide_name,
                        status_file=status_file,
                    )
                )
    return tuple(artifacts)

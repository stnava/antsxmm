from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .bids_entities import parse_entities
from .execution_plan import modality_from_path
from .run_id import normalize_run_id


def _extract_run(path: Path) -> str:
    entities = parse_entities(path.name)
    return normalize_run_id(entities.get('run'))


def predict_tree(subject_dir: str | Path) -> tuple[str, str, Dict[str, List[Tuple[str, str]]]]:
    """Predict the antsxmm output tree for a single subject directory.

    Parameters
    ----------
    subject_dir:
        Path like: <bids>/<project>/<subject> (i.e. contains ses-* children).

    Returns
    -------
    (project, subject, tree)
        tree maps session name -> list of (modality, run_id) tuples.
    """
    subject_dir = Path(subject_dir)

    if len(subject_dir.parts) < 3:
        raise ValueError(f"subject_dir must be a BIDS subject directory, got: {subject_dir}")

    project = subject_dir.parts[-2]
    subject = subject_dir.name

    sessions = sorted(subject_dir.glob("ses-*"))

    tree: Dict[str, List[Tuple[str, str]]] = {}

    for ses in sessions:
        runs_set: set[Tuple[str, str]] = set()

        for f in ses.rglob("*.nii.gz"):
            modality = modality_from_path(f.name)
            if modality is None:
                continue
            run_id = _extract_run(f)
            runs_set.add((modality, run_id))
            if modality == "T1w":
                runs_set.add(("T1wHierarchical", run_id))

        tree[ses.name] = sorted(runs_set, key=lambda x: (x[1], x[0]))

    return project, subject, tree

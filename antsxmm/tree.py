from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple


def _extract_run(path: Path) -> str:
    m = re.search(r"run-(\d+)", path.name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    m = re.search(r"(?:^|_)(?:r)(\d+)(?:[_.]|$)", path.name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    return "run-01"


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

    # Expect BIDS-ish: <bids>/<project>/<subject>
    if len(subject_dir.parts) < 3:
        raise ValueError(f"subject_dir must be a BIDS subject directory, got: {subject_dir}")

    project = subject_dir.parts[-2]
    subject = subject_dir.name

    sessions = sorted(subject_dir.glob("ses-*"))

    tree: Dict[str, List[Tuple[str, str]]] = {}

    for ses in sessions:
        runs: List[Tuple[str, str]] = []

        for f in ses.rglob("*.nii.gz"):
            run_id = _extract_run(f)

            if "T1w" in f.name:
                runs.append(("T1w", run_id))
                runs.append(("T1wHierarchical", run_id))

            if "_dwi" in f.name:
                runs.append(("DTI", run_id))

            if "_bold" in f.name:
                runs.append(("rsfMRI", run_id))

            if "_asl" in f.name:
                runs.append(("perf", run_id))

            # FLAIR (passed to antspymm as flair_filename)
            if "FLAIR" in f.name or "_flair" in f.name.lower():
                runs.append(("T2Flair", run_id))

            # PET (passed to antspymm as pet3d_filename)
            if "_pet" in f.name.lower() or f.parent.name.lower() == "pet":
                runs.append(("pet3d", run_id))

        tree[ses.name] = runs

    return project, subject, tree

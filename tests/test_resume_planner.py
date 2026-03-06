import json
from pathlib import Path

import pandas as pd


def test_run_study_resume_skips_unchanged_success(tmp_path, monkeypatch):
    """If status=success and input fingerprint matches, run_study should skip."""
    from antsxmm.pipeline import run_study
    from antsxmm.core import compute_input_fingerprint

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    # Create a fake T1 and sidecar
    t1 = tmp_path / "sub-01_T1w_run-01.nii.gz"
    t1.write_bytes(b"fake")
    (tmp_path / "sub-01_T1w_run-01.json").write_text("{}", encoding="utf-8")

    row = {
        "subjectID": "sub-01",
        "date": "ses-01",
        # Use scalar form to avoid pandas object normalization edge-cases.
        "t1_filename": str(t1),
        "session_path": str(tmp_path),
    }

    def fake_parse(_):
        return pd.DataFrame([row])

    monkeypatch.setattr("antsxmm.pipeline.parse_antsxbids_layout", fake_parse)

    # Pre-write a success status with matching fingerprint
    out_root = tmp_path / "out"
    status_path = out_root / "Project" / "sub-01" / "ses-01" / ".antsxmm_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    fp = compute_input_fingerprint(row)
    status_path.write_text(
        json.dumps({"success": True, "input_fingerprint": fp}),
        encoding="utf-8",
    )
    markers = [
        out_root / "Project" / "sub-01" / "ses-01" / "T1w" / "run-01" / "Project+sub-01+ses-01+T1w+run-01+mmwide.csv",
        out_root / "Project" / "sub-01" / "ses-01" / "T1wHierarchical" / "run-01" / "Project+sub-01+ses-01+T1wHierarchical+run-01+mmwide.csv",
    ]
    for marker in markers:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")

    def should_not_run(*args, **kwargs):
        raise AssertionError("process_session should not be called when resume skips")

    monkeypatch.setattr("antsxmm.pipeline.process_session", should_not_run)

    failures = run_study(
        bids_dir=str(bids_dir),
        output_dir=str(out_root),
        project="Project",
        resume=True,
        force=False,
        rerun_failed=False,
        dry_run=False,
        verbose=False,
    )

    assert failures == []

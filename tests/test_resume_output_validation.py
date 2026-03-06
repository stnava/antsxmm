import json

import pandas as pd


def test_run_study_reruns_when_success_status_is_stale_and_outputs_missing(tmp_path, monkeypatch):
    from antsxmm.pipeline import run_study
    from antsxmm.core import compute_input_fingerprint

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    t1 = tmp_path / "sub-01_T1w_run-01.nii.gz"
    perf = tmp_path / "sub-01_asl_run-01.nii.gz"
    t1.write_bytes(b"fake")
    perf.write_bytes(b"fake")

    row = {
        "subjectID": "sub-01",
        "date": "ses-01",
        "t1_filename": str(t1),
        "perf_filename": str(perf),
        "session_path": str(tmp_path),
    }

    monkeypatch.setattr("antsxmm.pipeline.parse_antsxbids_layout", lambda _: pd.DataFrame([row]))

    out_root = tmp_path / "out"
    status_path = out_root / "Project" / "sub-01" / "ses-01" / ".antsxmm_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    fp = compute_input_fingerprint(row)
    status_path.write_text(
        json.dumps({"success": True, "input_fingerprint": fp}),
        encoding="utf-8",
    )

    # Only structural marker exists; perf outputs are missing, so resume must rerun.
    t1_marker = out_root / "Project" / "sub-01" / "ses-01" / "T1w" / "run-01" / "Project+sub-01+ses-01+T1w+run-01+mmwide.csv"
    t1_marker.parent.mkdir(parents=True, exist_ok=True)
    t1_marker.write_text("ok", encoding="utf-8")

    called = []

    def fake_process_session(*_args, **_kwargs):
        called.append(True)
        return {"success": True}

    monkeypatch.setattr("antsxmm.pipeline.process_session", fake_process_session)

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
    assert len(called) == 1

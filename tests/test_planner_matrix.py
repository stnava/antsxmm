import json

import pandas as pd


def test_run_study_inputs_changed_triggers_rerun(tmp_path, monkeypatch):
    """If status=success but fingerprint differs, run_study must re-run."""
    from antsxmm.pipeline import run_study

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    row = {
        "subjectID": "sub-01",
        "date": "ses-01",
        "t1_filename": str(tmp_path / "sub-01_T1w_run-01.nii.gz"),
        "session_path": str(tmp_path),
    }
    (tmp_path / "sub-01_T1w_run-01.nii.gz").write_bytes(b"fake")

    monkeypatch.setattr("antsxmm.pipeline.parse_antsxbids_layout", lambda _: pd.DataFrame([row]))

    # Pre-write a success status with a different fingerprint
    out_root = tmp_path / "out"
    status_path = out_root / "Project" / "sub-01" / "ses-01" / ".antsxmm_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"success": True, "input_fingerprint": {"algo": "sha256", "hash": "OLD"}}),
        encoding="utf-8",
    )

    # Force the computed fingerprint to be different
    monkeypatch.setattr(
        "antsxmm.pipeline.compute_input_fingerprint",
        lambda *_args, **_kwargs: {"algo": "sha256", "hash": "NEW"},
    )

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


def test_run_study_rerun_failed_only_runs_failed_or_missing(tmp_path, monkeypatch):
    """--rerun-failed should only run sessions with no status or success=false."""
    from antsxmm.pipeline import run_study

    bids_dir = tmp_path / "bids"
    bids_dir.mkdir()

    row_ok = {
        "subjectID": "sub-01",
        "date": "ses-01",
        "t1_filename": str(tmp_path / "sub-01_T1w_run-01.nii.gz"),
        "session_path": str(tmp_path),
    }
    row_bad = {
        "subjectID": "sub-02",
        "date": "ses-01",
        "t1_filename": str(tmp_path / "sub-02_T1w_run-01.nii.gz"),
        "session_path": str(tmp_path),
    }
    (tmp_path / "sub-01_T1w_run-01.nii.gz").write_bytes(b"fake")
    (tmp_path / "sub-02_T1w_run-01.nii.gz").write_bytes(b"fake")

    monkeypatch.setattr("antsxmm.pipeline.parse_antsxbids_layout", lambda _: pd.DataFrame([row_ok, row_bad]))

    out_root = tmp_path / "out"
    # sub-01 already success
    p1 = out_root / "Project" / "sub-01" / "ses-01" / ".antsxmm_status.json"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps({"success": True}), encoding="utf-8")

    # sub-02 previously failed
    p2 = out_root / "Project" / "sub-02" / "ses-01" / ".antsxmm_status.json"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text(json.dumps({"success": False}), encoding="utf-8")

    monkeypatch.setattr(
        "antsxmm.pipeline.compute_input_fingerprint",
        lambda *_args, **_kwargs: {"algo": "sha256", "hash": "X"},
    )

    ran = []

    def fake_process_session(session_data, **_kwargs):
        ran.append((session_data["subjectID"], session_data["date"]))
        return {"success": True}

    monkeypatch.setattr("antsxmm.pipeline.process_session", fake_process_session)

    failures = run_study(
        bids_dir=str(bids_dir),
        output_dir=str(out_root),
        project="Project",
        resume=True,
        force=False,
        rerun_failed=True,
        dry_run=False,
        verbose=False,
    )

    assert failures == []
    assert ran == [("sub-02", "ses-01")]

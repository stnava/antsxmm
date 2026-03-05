import json


def test_process_session_writes_failure_status_on_exception(tmp_path, monkeypatch):
    """If antspymm raises, we should persist .antsxmm_status.json with success=false."""
    from antsxmm.session import process_session
    import antsxmm.session as session_mod

    # Minimal valid inputs
    t1 = tmp_path / "sub-01_T1w_run-01.nii.gz"
    t1.write_bytes(b"fake")

    row = {
        "subjectID": "sub-01",
        "date": "ses-01",
        "t1_filename": str(t1),
        "session_path": str(tmp_path),
    }

    # Provide minimal antspymm surface so process_session reaches mm_csv.
    session_mod.antspymm.generate_mm_dataframe = lambda **_kwargs: __import__("pandas").DataFrame([{ "ok": 1 }])
    session_mod.antspymm.get_data = lambda *_args, **_kwargs: None

    # Force antspymm failure
    def boom(*_args, **_kwargs):
        raise RuntimeError("mm_csv failed")

    session_mod.antspymm.mm_csv = boom

    out_root = tmp_path / "out"
    res = process_session(
        row,
        output_root=str(out_root),
        project_id="Project",
        build_wide_table=False,
        write_input_manifest=False,
        verbose=False,
    )

    assert res["success"] is False

    status_path = out_root / "Project" / "sub-01" / "ses-01" / ".antsxmm_status.json"
    assert status_path.exists()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["success"] is False
    assert "mm_csv failed" in (status.get("error") or "")

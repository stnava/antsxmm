def test_fingerprint_changes_when_sidecar_changes(tmp_path):
    from antsxmm.core import compute_input_fingerprint

    t1 = tmp_path / "sub-01_T1w_run-01.nii.gz"
    js = tmp_path / "sub-01_T1w_run-01.json"
    t1.write_bytes(b"nii")
    js.write_text("{}", encoding="utf-8")

    row = {
        "subjectID": "sub-01",
        "date": "ses-01",
        "t1_filename": str(t1),
        "session_path": str(tmp_path),
    }

    fp1 = compute_input_fingerprint(row)
    assert fp1["processable"] is True
    assert fp1["hash"]

    # Mutate sidecar content (size changes => fingerprint must change)
    js.write_text('{"EchoTime": 0.003}', encoding="utf-8")

    fp2 = compute_input_fingerprint(row)
    assert fp2["processable"] is True
    assert fp2["hash"]
    assert fp1["hash"] != fp2["hash"]

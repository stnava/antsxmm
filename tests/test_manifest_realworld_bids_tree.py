import json
import os
from pathlib import Path

import pytest

from .test_manifest_regressions import _DummyAntsPyMM, _touch


@pytest.fixture
def dummy_antspymm(monkeypatch):
    import antsxmm.core as core
    monkeypatch.setattr(core, "antspymm", _DummyAntsPyMM())
    return core


def _read_manifest(out_root: Path, project: str, sub: str, ses: str, sep: str = "+") -> dict:
    p = out_root / project / sub / ses / f"{project}{sep}{sub}{sep}{ses}{sep}mm_inputs.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _all_paths(obj):
    """Yield all string paths from a nested manifest structure."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _all_paths(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_paths(v)
        return


def test_manifest_contract_matches_realworld_bids_tree(tmp_path, dummy_antspymm, monkeypatch):
    """Contract test based on a real-world BIDS tree.

    Ensures manifests are internally consistent when the BIDS parser yields relative paths
    (e.g., 'BIDS/...') and truncation thresholds exceed available runs.
    """
    core = dummy_antspymm

    # Recreate the subset of the user's tree needed to exercise selection + manifest logic.
    bids_root = tmp_path / "BIDS"

    # breacher/sub-9162/ses-followup-day2 (1 T1, 1 DWI, 1 rsfMRI, 1 ASL)
    breacher = bids_root / "breacher" / "sub-9162" / "ses-followup-day2"
    t1 = _touch(str(breacher / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz"))
    dwi = _touch(str(breacher / "dwi" / "sub-9162_ses-followup-day2_dir-run-01_dwi.nii.gz"))
    _touch(str(breacher / "dwi" / "sub-9162_ses-followup-day2_dir-run-01_dwi.bval"))
    _touch(str(breacher / "dwi" / "sub-9162_ses-followup-day2_dir-run-01_dwi.bvec"))
    rsf = _touch(str(breacher / "func" / "sub-9162_ses-followup-day2_task-rest_run-01_bold.nii.gz"))
    asl = _touch(str(breacher / "perf" / "sub-9162_ses-followup-day2_run-01_asl.nii.gz"))

    out_root = tmp_path / "out"

    # Simulate a common parser behavior: relative paths rooted at the project working directory.
    monkeypatch.chdir(tmp_path)

    rel_t1 = os.path.relpath(t1, start=tmp_path)
    rel_dwi = os.path.relpath(dwi, start=tmp_path)
    rel_rsf = os.path.relpath(rsf, start=tmp_path)
    rel_asl = os.path.relpath(asl, start=tmp_path)

    session_data = {
        "subjectID": "sub-9162",
        "sessionID": "ses-followup-day2",
        "session_path": str(breacher),
        "t1_filenames": [rel_t1],
        "flair_filenames": [],
        "t2w_filenames": [],
        "dti_filenames": [rel_dwi],
        "rsf_filenames": [rel_rsf],
        "nm_filenames": [],
        "perf_filename": rel_asl,
        "pet3d_filename": None,
        "perf_filenames": [rel_asl],
        "pet3d_filenames": [],
    }

    # The pipeline may raise later if ants/antspymm isn't installed; the manifest must still be valid.
    core.process_session(
        session_data=session_data,
        output_root=str(out_root),
        project_id="breacher",
        denoise=False,
        separator="+",
        verbose=False,
        write_input_manifest=True,
        dti_moco=True,
        denoise_dti=True,
        t1_run_match=None,
    )

    m = _read_manifest(out_root, "breacher", "sub-9162", "ses-followup-day2", sep="+")

    # 1) Truncation invariants: when discovered <= 2, nothing should be "truncated".
    assert len(m["discovered"]["rsf_filenames"]) == 1
    assert len(m["discovered"]["dti_filenames"]) == 1
    assert m["excluded"]["rsf_truncated"] == []
    assert m["excluded"]["dti_truncated"] == []

    # 2) Selection invariants: selected T1 must not be listed as excluded.
    assert m["excluded"]["t1_not_selected"] == []

    # 3) Path consistency: all discovered/used/processed paths must be absolute realpaths.
    all_paths = list(_all_paths({
        "discovered": m["discovered"],
        "used_inputs": m["used_inputs"],
        "nifti_inputs_that_will_be_processed": m["nifti_inputs_that_will_be_processed"],
        "excluded": m["excluded"],
    }))
    assert all_paths, "expected some paths in the manifest"
    assert all(os.path.isabs(p) for p in all_paths), "manifest must not mix relative paths"
    assert all(p == os.path.realpath(p) for p in all_paths), "manifest paths must be normalized via realpath"

    # 4) Accounting invariants: excluded must be disjoint from used, and processed equals union of used.
    used = set(_all_paths(m["used_inputs"]))
    excluded = set(_all_paths(m["excluded"]))
    processed = set(m["nifti_inputs_that_will_be_processed"])
    assert used.isdisjoint(excluded)

    # Processed list should equal used inputs (flattened) minus any None values.
    assert processed == used
def test_manifest_contract_fpa_asl_directory_is_treated_as_perf(tmp_path, dummy_antspymm, monkeypatch):
    """Contract test for BIDS/FPA layout where ASL lives under an 'asl/' directory.

    The pipeline must treat ASL as perfusion (perf) even if the parser does not populate perf_filenames.
    """
    core = dummy_antspymm

    bids_root = tmp_path / "BIDS"
    ses = bids_root / "FPA" / "sub-BLAST048MRI" / "ses-01"

    t1 = _touch(str(ses / "anat" / "sub-BLAST048MRI_ses-01_run-001_T1w.nii.gz"))
    flair = _touch(str(ses / "anat" / "sub-BLAST048MRI_ses-01_run-001_FLAIR.nii.gz"))
    t2w = _touch(str(ses / "anat" / "sub-BLAST048MRI_ses-01_run-001_T2w.nii.gz"))
    dwi_ap = _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-AP_dwi.nii.gz"))
    _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-AP_dwi.bval"))
    _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-AP_dwi.bvec"))
    dwi_pa = _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-PA_dwi.nii.gz"))
    _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-PA_dwi.bval"))
    _touch(str(ses / "dwi" / "sub-BLAST048MRI_ses-01_run-001_dir-PA_dwi.bvec"))
    rsf = _touch(str(ses / "func" / "sub-BLAST048MRI_ses-01_task-rest_run-001_bold.nii.gz"))

    # ASL stored under 'asl/' (not 'perf/')
    asl = _touch(str(ses / "asl" / "sub-BLAST048MRI_ses-01_run-001_asl.nii.gz"))
    _touch(str(ses / "asl" / "sub-BLAST048MRI_ses-01_run-001_asl.json"))

    out_root = tmp_path / "out"
    monkeypatch.chdir(tmp_path)

    rel_t1 = os.path.relpath(t1, start=tmp_path)
    rel_flair = os.path.relpath(flair, start=tmp_path)
    rel_t2w = os.path.relpath(t2w, start=tmp_path)
    rel_dwi_ap = os.path.relpath(dwi_ap, start=tmp_path)
    rel_dwi_pa = os.path.relpath(dwi_pa, start=tmp_path)
    rel_rsf = os.path.relpath(rsf, start=tmp_path)

    session_data = {
        "subjectID": "sub-BLAST048MRI",
        "sessionID": "ses-01",
        "session_path": str(ses),
        "t1_filenames": [rel_t1],
        "flair_filenames": [rel_flair],
        "t2w_filenames": [rel_t2w],
        "dti_filenames": [rel_dwi_ap, rel_dwi_pa],
        "rsf_filenames": [rel_rsf],
        "nm_filenames": [],
        # IMPORTANT: Simulate the parser not populating perf fields for 'asl/' layouts.
        "perf_filename": None,
        "perf_filenames": [],
        "pet3d_filename": None,
        "pet3d_filenames": [],
    }

    core.process_session(
        session_data=session_data,
        output_root=str(out_root),
        project_id="FPA",
        denoise=False,
        separator="+",
        verbose=False,
        write_input_manifest=True,
        dti_moco=True,
        denoise_dti=True,
        t1_run_match=None,
    )

    m = _read_manifest(out_root, "FPA", "sub-BLAST048MRI", "ses-01", sep="+")

    # Must discover ASL under asl/ as perfusion.
    assert m["discovered"]["perf_filenames"], "expected ASL to be discovered as perfusion"
    assert os.path.realpath(asl) in m["discovered"]["perf_filenames"]

    # Must select the ASL as perf (either perf_filename or in processed list).
    used_perf = m["used_inputs"].get("perf_filename")
    assert used_perf == os.path.realpath(asl)
    assert os.path.realpath(asl) in m["nifti_inputs_that_will_be_processed"]

    # Manifest invariants still hold.
    used = set(_all_paths(m["used_inputs"]))
    excluded = set(_all_paths(m["excluded"]))
    assert used.isdisjoint(excluded)

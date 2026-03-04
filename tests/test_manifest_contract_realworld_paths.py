import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from antsxmm.core import process_session


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_input_manifest_contract_no_relative_absolute_mixing_and_exclusions(tmp_path, monkeypatch):
    """Regression/contract test for manifest invariants.

    Historically, manifests could mix relative and absolute paths, which then caused
    bogus exclusions (e.g., marking the only DTI/rsf file as truncated).

    This test simulates a real-world BIDS row that uses relative paths and asserts
    the written manifest is internally consistent.
    """

    # Arrange a minimal real-world-ish BIDS layout using RELATIVE paths.
    bids_root = tmp_path / "BIDS" / "breacher" / "sub-9162" / "ses-followup-day2"
    (bids_root / "anat").mkdir(parents=True)
    (bids_root / "dwi").mkdir(parents=True)
    (bids_root / "func").mkdir(parents=True)
    (bids_root / "perf").mkdir(parents=True)

    rel_t1 = "BIDS/breacher/sub-9162/ses-followup-day2/anat/sub-9162_ses-followup-day2_T1w.nii.gz"
    rel_dti = "BIDS/breacher/sub-9162/ses-followup-day2/dwi/sub-9162_ses-followup-day2_dir-run-01_dwi.nii.gz"
    rel_rsf = "BIDS/breacher/sub-9162/ses-followup-day2/func/sub-9162_ses-followup-day2_task-rest_run-01_bold.nii.gz"
    rel_perf = "BIDS/breacher/sub-9162/ses-followup-day2/perf/sub-9162_ses-followup-day2_run-01_asl.nii.gz"

    # Touch files at their relative locations.
    (tmp_path / rel_t1).touch()
    (tmp_path / rel_dti).touch()
    (tmp_path / rel_rsf).touch()
    (tmp_path / rel_perf).touch()

    # Ensure os.path.exists() works for the relative paths.
    monkeypatch.chdir(tmp_path)

    session_data = {
        "subjectID": "sub-9162",
        "date": "ses-followup-day2",
        "session_path": "BIDS/breacher/sub-9162/ses-followup-day2",
        "t1_filenames": [rel_t1],
        "dti_filenames": [rel_dti],
        "rsf_filenames": [rel_rsf],
        "perf_filenames": [rel_perf],
        "nm_filenames": [],
        "flair_filenames": [],
        "t2w_filenames": [],
        "pet3d_filenames": [],
    }

    out_root = tmp_path / "out"

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", create=True) as mock_gen, \
        patch("antsxmm.core.antspymm.mm_csv", create=True), \
        patch("antsxmm.core.antspymm.get_data", return_value=None, create=True):
        mock_gen.return_value = pd.DataFrame({"dtid1": ["placeholder"]})

        # Act
        process_session(
            session_data,
            str(out_root),
            project_id="breacher",
            verbose=False,
            build_wide_table=False,
            write_input_manifest=True,
        )

    manifest_path = (
        out_root
        / "breacher"
        / "sub-9162"
        / "ses-followup-day2"
        / "breacher_sub-9162_ses-followup-day2_mm_inputs.json"
    )
    assert manifest_path.exists(), f"expected manifest at {manifest_path}"

    manifest = _load_json(manifest_path)

    # Assert: no bogus truncation when discovered <= truncate_N
    assert len(manifest["discovered"]["dti_filenames"]) == 1
    assert manifest["excluded"]["dti_truncated"] == []
    assert len(manifest["discovered"]["rsf_filenames"]) == 1
    assert manifest["excluded"]["rsf_truncated"] == []

    # Assert: the only T1 cannot be simultaneously used and not-selected
    assert manifest["excluded"]["t1_not_selected"] == []

    # Assert: all manifest paths are normalized consistently (absolute realpaths)
    def _all_paths(obj):
        if obj is None:
            return []
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, str)]
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                out.extend(_all_paths(v))
            return out
        return []

    all_paths = _all_paths(
        {
            "discovered": manifest["discovered"],
            "used_inputs": manifest["used_inputs"],
            "excluded": manifest["excluded"],
            "nifti_inputs_that_will_be_processed": manifest["nifti_inputs_that_will_be_processed"],
        }
    )
    assert all(os.path.isabs(p) for p in all_paths), "manifest must not mix relative paths"

    # Assert: excluded ∩ used = ∅
    used_set = set(
        [p for p in _all_paths(manifest["used_inputs"]) if p]
    )
    excluded_set = set(
        [p for p in _all_paths(manifest["excluded"]) if p]
    )
    assert used_set.isdisjoint(excluded_set)

    # Assert: nifti_inputs_that_will_be_processed is exactly the union of used NIfTIs
    union_used = sorted(used_set)
    assert sorted(manifest["nifti_inputs_that_will_be_processed"]) == union_used

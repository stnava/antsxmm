from pathlib import Path
from unittest.mock import patch

import pandas as pd

from antsxmm.core import process_session


def test_non_t1_modality_output_dirs_are_normalized_to_run_id(tmp_path, mock_session_data):
    """Regression: antspymm may create non-T1 modality output dirs named after the full input stem.

    Expected stable structure for downstream tools:
      <out>/<project>/<sub>/<ses>/<modality>/<run-id>/...
    """

    out_root = tmp_path / "pymm"
    project = "FPA"
    sub = "sub-BLAST034"
    ses = "ses-01"

    # Use filenames that carry run-001 in the stem (matches the reported failure).
    # We only need T1 + (flair, pet) to exercise the rename logic.
    bids_root = tmp_path / "BIDS" / project / sub / ses
    (bids_root / "anat").mkdir(parents=True)
    t1 = bids_root / "anat" / f"{sub}_{ses}_T1w_run-001.nii.gz"
    t1.touch()
    flair = bids_root / "anat" / f"{sub}_{ses}_T2Flair_run-001.nii.gz"
    flair.touch()
    pet = bids_root / "pet" / f"{sub}_{ses}_pet3d_run-001.nii.gz"
    pet.parent.mkdir(parents=True)
    pet.touch()

    session_data = dict(mock_session_data)
    session_data.update(
        {
            "subjectID": sub,
            "date": ses,
            "session_path": str(bids_root),
            "t1_filename": str(t1),
            "t1_filenames": [str(t1)],
            "flair_filename": str(flair),
            "flair_filenames": [str(flair)],
            "pet3d_filename": str(pet),
            "pet3d_filenames": [str(pet)],
            "rsf_filenames": [],
            "dti_filenames": [],
            "nm_filenames": [],
            "perf_filenames": [],
        }
    )

    # antspymm stubs: create the *erroneous* directory names inside the real output tree.
    def _fake_mm_csv(*args, **kwargs):
        session_dir = Path(out_root) / project / sub / ses
        # Mimic the reported antspymm behavior:
        #   T2Flair/<full-stem>/...
        #   pet3d/<full-stem>/...
        t2_bad = session_dir / "T2Flair" / flair.stem
        pet_bad = session_dir / "pet3d" / pet.stem
        t2_bad.mkdir(parents=True, exist_ok=True)
        pet_bad.mkdir(parents=True, exist_ok=True)
        (t2_bad / "dummy.txt").write_text("x", encoding="utf-8")
        (pet_bad / "dummy.txt").write_text("x", encoding="utf-8")

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", return_value=pd.DataFrame({"A": [1]}), create=True), \
        patch("antsxmm.core.antspymm.mm_csv", side_effect=_fake_mm_csv, create=True):
        res = process_session(
            session_data,
            output_root=str(out_root),
            project_id=project,
            separator="+",
            build_wide_table=False,
            write_input_manifest=False,
            verbose=False,
        )

    assert res["success"] is True

    session_dir = Path(out_root) / project / sub / ses

    # The fix: bad dir renamed to run-001.
    assert (session_dir / "T2Flair" / "run-001").is_dir()
    assert (session_dir / "pet3d" / "run-001").is_dir()

    # Old names should not remain.
    assert not (session_dir / "T2Flair" / flair.stem).exists()
    assert not (session_dir / "pet3d" / pet.stem).exists()

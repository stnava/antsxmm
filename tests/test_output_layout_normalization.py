
import pytest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

from antsxmm.core import process_session


def _mkfile(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_output_layout_and_prefix_are_normalized_for_pet_and_t1w(tmp_path):
    """
    Real-world regression:
      - pet3d files sometimes include a verbose run segment like:
          ...+pet3d+sub-..._pet3d_run-001_run-001+mmwide.csv
        but should be:
          ...+pet3d+run-001+mmwide.csv

      - T1w sometimes emits an extra folder:
          T1w/sub-..._T1w_run-001/
        whose contents should be merged into:
          T1w/run-001/
    """
    output_root = tmp_path / "pymm"
    project_id = "FPA"
    sub_id = "sub-BLAST034"
    date_id = "ses-01"

    # Minimal on-disk inputs expected by process_session staging
    t1 = tmp_path / "BIDS" / "FPA" / sub_id / date_id / "anat" / f"{sub_id}_{date_id}_run-001_T1w.nii.gz"
    pet = tmp_path / "BIDS" / "FPA" / sub_id / date_id / "pet" / f"{sub_id}_{date_id}_pet3d_run-001.nii.gz"
    _mkfile(t1)
    _mkfile(pet)

    session_data = {
        "projectID": project_id,
        "subjectID": sub_id,
        "date": date_id,
        "t1_filename": str(t1),
        "pet3d_filename": str(pet),
    }

    # generate_mm_dataframe returns *a* DataFrame; mm_csv is mocked to write the problematic tree
    def fake_generate_mm_dataframe(**kwargs):
        return pd.DataFrame([{
            "projectID": project_id,
            "subjectID": sub_id,
            "date": date_id,
            "imageUniqueID": "uid",
            "modality": "T1w",
            "t1_filename": str(t1),
            "pet3d_filename": str(pet),
        }])

    def fake_mm_csv(df, **kwargs):
        session_out = Path(output_root) / project_id / sub_id / date_id

        # pet3d: verbose run segment in both dir name and filename
        bad_pet_dir = session_out / "pet3d" / f"{sub_id}_{date_id}_pet3d_run-001"
        _mkfile(bad_pet_dir / f"{project_id}+{sub_id}+{date_id}+pet3d+{sub_id}_{date_id}_pet3d_run-001_run-001+mmwide.csv")

        # T1w: canonical run dir plus an extra verbose dir that should be merged
        good_t1_dir = session_out / "T1w" / "run-001"
        bad_t1_dir = session_out / "T1w" / f"{sub_id}_{date_id}_T1w_run-001"
        _mkfile(good_t1_dir / f"{project_id}+{sub_id}+{date_id}+T1w+run-001+syn0GenericAffine.mat")
        _mkfile(bad_t1_dir / f"{project_id}+{sub_id}+{date_id}+T1w+run-001+brainextraction.png")

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", create=True, side_effect=fake_generate_mm_dataframe), \
         patch("antsxmm.core.antspymm.mm_csv", create=True, side_effect=fake_mm_csv), \
         patch("antsxmm.core.antspymm.get_data", create=True, return_value=None):
        process_session(
            session_data,
            output_root=str(output_root),
            project_id=project_id,
            build_wide_table=False,
            write_input_manifest=False,
            verbose=False,
        )

    session_out = output_root / project_id / sub_id / date_id

    # pet3d: dir normalized + run segment rewritten
    assert (session_out / "pet3d" / "run-001").is_dir()
    assert (session_out / "pet3d" / "run-001" / f"{project_id}+{sub_id}+{date_id}+pet3d+run-001+mmwide.csv").is_file()

    # T1w: merged into run-001; verbose dir removed
    assert (session_out / "T1w" / "run-001" / f"{project_id}+{sub_id}+{date_id}+T1w+run-001+syn0GenericAffine.mat").is_file()
    assert (session_out / "T1w" / "run-001" / f"{project_id}+{sub_id}+{date_id}+T1w+run-001+brainextraction.png").is_file()
    assert not (session_out / "T1w" / f"{sub_id}_{date_id}_T1w_run-001").exists()

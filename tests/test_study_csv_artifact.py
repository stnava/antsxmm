from pathlib import Path
from unittest.mock import patch

import pandas as pd

from antsxmm.core import process_session


def test_study_csv_is_written(tmp_path):
    out_root = tmp_path / "out"
    t1 = tmp_path / "sub-001_ses-20230101_T1w.nii.gz"
    t1.touch()

    session_data = {
        "subjectID": "sub-001",
        "date": "ses-20230101",
        "t1_filenames": [str(t1)],
        # keep everything else empty to avoid optional modality branches
        "flair_filenames": [],
        "t2w_filenames": [],
        "rsf_filenames": [],
        "dti_filenames": [],
        "nm_filenames": [],
        "perf_filenames": [],
        "pet3d_filenames": [],
    }

    df = pd.DataFrame({"A": [1]})

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", return_value=df, create=True), \
         patch("antsxmm.core.antspymm.mm_csv", create=True):
        res = process_session(
            session_data,
            str(out_root),
            project_id="Project",
            verbose=False,
            write_input_manifest=False,
        )

    assert res["success"] is True
    session_dir = out_root / "Project" / "sub-001" / "ses-20230101"
    study_csv = session_dir / "Project+sub-001+ses-20230101+study.csv"
    assert study_csv.exists(), f"Expected study.csv at {study_csv}"

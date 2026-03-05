from pathlib import Path
from unittest.mock import patch

import pandas as pd

from antsxmm.core import process_session


def test_numeric_index_output_dirs_are_mapped_to_input_run_ids_and_t1_defaults_to_run01(tmp_path):
    """Regression: antspymm may emit modality outputs under numeric index dirs (e.g. '000')
    and/or verbose stems (e.g. '<stem>_DTI_000'). We must normalize to the input run-id.

    Also, when T1w lacks an explicit run label, we should default to 'run-01' (not '000').
    """
    out_root = tmp_path / "pymm"
    project = "breacher"
    sub = "sub-9162"
    ses = "ses-followup-day2"

    bids_root = tmp_path / "BIDS" / project / sub / ses
    (bids_root / "anat").mkdir(parents=True)
    (bids_root / "dwi").mkdir(parents=True)
    (bids_root / "perf").mkdir(parents=True)

    # T1w WITHOUT run in filename
    t1 = bids_root / "anat" / f"{sub}_{ses}_T1w.nii.gz"
    t1.touch()

    # DWI WITH run-01 in stem (dir-run-01)
    dwi = bids_root / "dwi" / f"{sub}_{ses}_dir-run-01_dwi.nii.gz"
    dwi.touch()
    (bids_root / "dwi" / f"{sub}_{ses}_dir-run-01_dwi.bval").touch()
    (bids_root / "dwi" / f"{sub}_{ses}_dir-run-01_dwi.bvec").touch()

    # ASL WITH run-01 in stem
    asl = bids_root / "perf" / f"{sub}_{ses}_run-01_asl.nii.gz"
    asl.touch()

    session_data = {
        "subjectID": sub,
        "date": ses,
        "session_path": str(bids_root),
        "t1_filename": str(t1),
        "t1_filenames": [str(t1)],
        "dti_filenames": [str(dwi)],
        "rsf_filenames": [],
        "nm_filenames": [],
        "perf_filename": str(asl),
        "perf_filenames": [str(asl)],
        "flair_filenames": [],
        "t2w_filenames": [],
        "pet3d_filenames": [],
    }

    def _fake_mm_csv(*args, **kwargs):
        session_dir = Path(out_root) / project / sub / ses

        # Mimic the problematic antspymm behavior from the user's output:
        # - DTI/sub-..._DTI_000
        # - perf/sub-..._perf_000
        # - T1w has BOTH '000' and 'sub-..._T1w_000'
        dti_bad = session_dir / "DTI" / f"{sub}_{ses}_DTI_000"
        perf_bad = session_dir / "perf" / f"{sub}_{ses}_perf_000"
        t1_bad_a = session_dir / "T1w" / "000"
        t1_bad_b = session_dir / "T1w" / f"{sub}_{ses}_T1w_000"

        for p in (dti_bad, perf_bad, t1_bad_a, t1_bad_b):
            p.mkdir(parents=True, exist_ok=True)
            (p / "dummy.txt").write_text("x", encoding="utf-8")

        # Also mimic file prefixes that embed the verbose run segment (5th '+' segment)
        # so rename_prefix_run_segment() has something to act on.
        (dti_bad / f"{project}+{sub}+{ses}+DTI+{sub}_{ses}_DTI_000_000+b0avg.nii.gz").write_text("x", encoding="utf-8")
        (perf_bad / f"{project}+{sub}+{ses}+perf+{sub}_{ses}_perf_000_000+cbf.nii.gz").write_text("x", encoding="utf-8")
        (t1_bad_b / f"{project}+{sub}+{ses}+T1w+000+kk_norm.nii.gz").write_text("x", encoding="utf-8")

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", return_value=pd.DataFrame({"A": [1]}), create=True), \
        patch("antsxmm.core.antspymm.mm_csv", side_effect=_fake_mm_csv, create=True):
        res = process_session(
            session_data=session_data,
            output_root=str(out_root),
            project_id=project,
            separator="+",
            build_wide_table=False,
            write_input_manifest=False,
            verbose=False,
        )

    assert res["success"] is True

    session_dir = Path(out_root) / project / sub / ses

    # Expect stable run-id dirs derived from input.
    assert (session_dir / "DTI" / "run-01").is_dir()
    assert (session_dir / "perf" / "run-01").is_dir()
    assert (session_dir / "T1w" / "run-01").is_dir()

    # Old dirs should be gone.
    assert not (session_dir / "DTI" / f"{sub}_{ses}_DTI_000").exists()
    assert not (session_dir / "perf" / f"{sub}_{ses}_perf_000").exists()
    assert not (session_dir / "T1w" / "000").exists()
    assert not (session_dir / "T1w" / f"{sub}_{ses}_T1w_000").exists()

    # Prefix run segment should be rewritten to run-01 for the files we created.
    dti_files = list((session_dir / "DTI" / "run-01").glob("*.nii.gz"))
    assert any("+DTI+run-01+" in f.name for f in dti_files)
    perf_files = list((session_dir / "perf" / "run-01").glob("*.nii.gz"))
    assert any("+perf+run-01+" in f.name for f in perf_files)


from antsxmm.core import _extract_run_id_from_filename


def test_run_id_extraction_present():
    f = "sub-01_ses-01_run-02_T1w.nii.gz"
    assert _extract_run_id_from_filename(f) == "run-02"


def test_run_id_extraction_missing():
    f = "sub-01_ses-01_T1w.nii.gz"
    assert _extract_run_id_from_filename(f) == "run-01"

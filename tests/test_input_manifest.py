import json
import pandas as pd
from unittest.mock import patch, MagicMock


def test_writes_input_manifest(mock_bids_structure, tmp_path):
    # Avoid running real ANTs/ANTsPyMM.
    mock_img = MagicMock()
    mock_img.__mul__.return_value = mock_img

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}), create=True), \
         patch("antsxmm.core.antspymm.get_data", return_value=None, create=True), \
         patch("antsxmm.core.antspymm.mm_csv", create=True), \
         patch("antsxmm.core.ants.image_read", return_value=mock_img, create=True), \
         patch("antsxmm.core.ants.crop_image", return_value=mock_img, create=True), \
         patch("antsxmm.core.ants.iMath", return_value=mock_img, create=True):

        from antsxmm.pipeline import run_study
        run_study(str(mock_bids_structure), str(tmp_path), "PROJ")

    manifest = tmp_path / "PROJ" / "sub-001" / "ses-20230101" / "PROJ+sub-001+ses-20230101+mm_inputs.json"
    assert manifest.exists(), "Expected per-session mm_inputs.json to be written"

    obj = json.loads(manifest.read_text())
    nifti_list = obj["nifti_inputs_that_will_be_processed"]

    # Exact inputs: includes T1 + FLAIR + rsfMRI + DWI(s) + NM + perf (ASL) + PET.
    assert any(p.endswith("_T1w.nii.gz") for p in nifti_list)
    assert any("FLAIR" in p for p in nifti_list)
    assert any(p.endswith("_bold.nii.gz") for p in nifti_list)
    assert any(p.endswith("_dwi.nii.gz") for p in nifti_list)
    assert any(p.endswith("_NM.nii.gz") for p in nifti_list)
    assert any(p.endswith("_asl.nii.gz") for p in nifti_list)
    assert any(p.endswith("_pet.nii.gz") for p in nifti_list)

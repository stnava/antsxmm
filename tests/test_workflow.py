import pytest
from antsxmm.pipeline import run_study
import pandas as pd

def test_whole_workflow(mock_bids_structure, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value="dummy")
    
    mock_img = mocker.MagicMock()
    mock_img.__mul__.return_value = mock_img
    mocker.patch("ants.image_read", return_value=mock_img)
    mocker.patch("ants.crop_image", return_value=mock_img)
    mocker.patch("ants.iMath", return_value=mock_img)
    
    mock_mm_csv = mocker.patch("antspymm.mm_csv")
    
    run_study(str(mock_bids_structure), str(tmp_path), "PROJ")
    assert mock_mm_csv.call_count == 1

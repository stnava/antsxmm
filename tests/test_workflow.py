import pytest
from antsxmm.pipeline import run_study
import pandas as pd
from unittest.mock import patch, MagicMock

def test_whole_workflow(mock_bids_structure, tmp_path):
    mock_img = MagicMock()
    mock_img.__mul__.return_value = mock_img

    with patch("antsxmm.core.antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}), create=True), \
         patch("antsxmm.core.antspymm.get_data", return_value="dummy", create=True), \
         patch("antsxmm.core.antspymm.mm_csv", create=True) as mock_mm_csv, \
         patch("antsxmm.core.ants.image_read", return_value=mock_img, create=True), \
         patch("antsxmm.core.ants.crop_image", return_value=mock_img, create=True), \
         patch("antsxmm.core.ants.iMath", return_value=mock_img, create=True):

        run_study(str(mock_bids_structure), str(tmp_path), "PROJ")
        assert mock_mm_csv.call_count == 1

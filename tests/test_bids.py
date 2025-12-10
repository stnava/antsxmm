import pytest
from antsxmm.bids import parse_antsxbids_layout

def test_parse_bids_valid(mock_bids_structure):
    df = parse_antsxbids_layout(mock_bids_structure)
    assert len(df) == 1
    row = df.iloc[0]
    assert "T1w.nii.gz" in row['t1_filename']
    # Check that it picked up the uppercase FLAIR
    assert "FLAIR.nii.gz" in row['flair_filename']

def test_parse_bids_no_dir():
    with pytest.raises(FileNotFoundError):
        parse_antsxbids_layout("non_existent_directory")

def test_parse_bids_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    df = parse_antsxbids_layout(tmp_path / "empty")
    assert df.empty

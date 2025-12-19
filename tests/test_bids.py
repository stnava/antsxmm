import pytest
from antsxmm.bids import parse_antsxbids_layout

def test_parse_bids_valid(mock_bids_structure):
    df = parse_antsxbids_layout(mock_bids_structure)
    assert len(df) == 1
    row = df.iloc[0]
    
    # Check that t1_filenames contains BOTH runs
    t1_list = row['t1_filenames']
    assert len(t1_list) == 2
    assert "r0001" in t1_list[0]
    assert "r0002" in t1_list[1]
    
    # We no longer expect a scalar 't1_filename' in the dataframe at this stage
    # It is derived in process_session
    assert 't1_filename' not in row or pd.isna(row['t1_filename'])

def test_parse_bids_no_dir():
    with pytest.raises(FileNotFoundError):
        parse_antsxbids_layout("non_existent_directory")

def test_parse_bids_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    df = parse_antsxbids_layout(tmp_path / "empty")
    assert df.empty
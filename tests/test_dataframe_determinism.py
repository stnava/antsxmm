from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def test_dataframe_generation_is_deterministic(tmp_path):
    t1 = _touch(tmp_path / 'sub-1' / 'ses-1' / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    session = {'subjectID': 'sub-1', 'sessionID': 'ses-1', 'session_path': str(tmp_path / 'sub-1' / 'ses-1'), 't1_filenames': [t1]}
    df1 = generate_xmm_dataframe(session, str(tmp_path / 'out'), 'Proj')
    df2 = generate_xmm_dataframe(session, str(tmp_path / 'out'), 'Proj')
    assert df1.to_dict(orient='records') == df2.to_dict(orient='records')

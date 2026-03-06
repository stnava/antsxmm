import types
from unittest.mock import patch

from antsxmm.session import process_session


class DummyAntsPyMM:
    def generate_mm_dataframe(self, **kwargs):
        import pandas as pd
        return pd.DataFrame([kwargs])

    def get_data(self, *args, **kwargs):
        return None

    def mm_csv(self, df, **kwargs):
        return {'ok': True}


class DummyAnts:
    pass


def test_process_session_validates_execution_plan_before_mm_csv(tmp_path, monkeypatch):
    t1 = tmp_path / 'sub-1_ses-1_run-001_T1w.nii.gz'
    t1.write_bytes(b'fake')

    session_data = {
        'subjectID': 'sub-1',
        'sessionID': 'ses-1',
        'date': 'ses-1',
        'session_path': str(tmp_path),
        't1_filenames': [str(t1)],
    }

    monkeypatch.setattr('antsxmm.session.antspymm', DummyAntsPyMM())
    monkeypatch.setattr('antsxmm.session.ants', DummyAnts())

    called = {'validate': 0, 'mm_csv': 0}

    def fake_validate(plan):
        called['validate'] += 1

    def fake_run(df, antspymm_module, **kwargs):
        called['mm_csv'] += 1
        return {'ok': True}

    with patch('antsxmm.session.validate_execution_plan', side_effect=fake_validate), \
         patch('antsxmm.session.run_xmm_mm_csv', side_effect=fake_run), \
         patch('antsxmm.core.sanitize_and_stage_file', side_effect=lambda f, *a, **k: (str(f) if f else None, None, 'run-01')), \
         patch('antsxmm.core.build_wide_table_from_mmwide', return_value=None):
        result = process_session(session_data, str(tmp_path / 'out'), project_id='Proj', verbose=False, build_wide_table=False)

    assert result['success'] is True
    assert called['validate'] == 1
    assert called['mm_csv'] == 1

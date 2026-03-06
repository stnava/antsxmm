import json
from pathlib import Path

from antsxmm.diagnostics import diagnose_session_inputs, summarize_input_diagnostics
from antsxmm.session import process_session


def test_diagnostics_reports_broken_symlink_for_declared_t1(tmp_path):
    target = tmp_path / 'missing_target.nii.gz'
    link = tmp_path / 'sub-01_T1w.nii.gz'
    link.symlink_to(target)

    session_data = {
        'subjectID': 'sub-01',
        'date': 'ses-01',
        't1_filenames': [str(link)],
    }

    plan = {
        'subjectID': 'sub-01',
        'sessionID': 'ses-01',
        'processable': False,
        'used': {},
        'selection_tracking': {},
    }
    diag = diagnose_session_inputs(session_data, plan=plan)

    t1 = diag['modalities']['t1']
    assert t1['declared_count'] == 1
    assert t1['usable_count'] == 0
    assert 'broken_symlink' in t1['failure_reasons']
    assert 'declared_T1w_candidates_not_usable' in diag['overall_failures']
    assert 'broken_symlink' in summarize_input_diagnostics(diag)


def test_process_session_writes_input_diagnostics_on_no_t1(tmp_path):
    session_data = {
        'subjectID': 'sub-01',
        'date': 'ses-01',
        't1_filenames': [],
    }

    result = process_session(session_data, str(tmp_path), verbose=False)
    assert result['success'] is False

    diag_path = tmp_path / 'ANTsX' / 'sub-01' / 'ses-01' / 'ANTsX_sub-01_ses-01_input_diagnostics.json'
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text())
    assert 'no_declared_T1w_candidates' in payload['overall_failures']
    assert 'no_declared_candidates' in payload['modalities']['t1']['failure_reasons']

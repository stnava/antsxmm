import json
from types import SimpleNamespace

from antsxmm.diagnostics import diagnose_bids_tree, diagnose_session_inputs
from antsxmm.session import process_session


def test_study_diagnostics_do_not_treat_missing_json_as_suspicious(tmp_path):
    bids_dir = tmp_path / 'BIDS' / 'ExpArt'
    anat = bids_dir / 'sub-FAIL03' / 'ses-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-FAIL03_ses-01_T1w.nii.gz').write_text('')

    diag = diagnose_bids_tree(bids_dir)

    assert diag['counts']['images_without_json'] == 1
    assert 'image_without_json' not in diag['suspicious_reasons']


def test_session_diagnostics_do_not_require_sidecars(tmp_path):
    img = tmp_path / 'sub-01_T1w.nii.gz'
    img.write_text('')
    diag = diagnose_session_inputs({'subjectID': 'sub-01', 'date': 'ses-01', 't1_filenames': [str(img)]}, plan={'used': {'t1_filename': str(img)}, 'selection_tracking': {}})
    inspected = diag['modalities']['t1']['inspected'][0]
    assert inspected['usable'] is True
    assert 'no_sidecars_found' not in inspected['reasons']



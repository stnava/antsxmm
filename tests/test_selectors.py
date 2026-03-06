import json
import os

from antsxmm.inputs import plan_session_inputs
from antsxmm.selectors import selector_for_modality, T1Selector, RestingStateSelector, DwiSelector


def _touch(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'')
    return path


def _write_json(path: str, obj: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    return path


def test_selector_registry_returns_explicit_named_selectors():
    assert isinstance(selector_for_modality('t1'), T1Selector)
    assert isinstance(selector_for_modality('rsf'), RestingStateSelector)
    assert isinstance(selector_for_modality('dti'), DwiSelector)


def test_selection_tracking_records_named_selector(tmp_path):
    ses = tmp_path / 'BIDS' / 'sub-1' / 'ses-1'
    t1 = _touch(str(ses / 'anat' / 'sub-1_ses-1_run-01_T1w.nii.gz'))
    rsf_lr = _touch(str(ses / 'func' / 'sub-1_ses-1_task-rest_dir-LR_bold.nii.gz'))
    rsf_rl = _touch(str(ses / 'func' / 'sub-1_ses-1_task-rest_dir-RL_bold.nii.gz'))
    dwi_lr = _touch(str(ses / 'dwi' / 'sub-1_ses-1_dir-LR_dwi.nii.gz'))
    dwi_rl = _touch(str(ses / 'dwi' / 'sub-1_ses-1_dir-RL_dwi.nii.gz'))
    perf = _touch(str(ses / 'perf' / 'sub-1_ses-1_run-01_asl.nii.gz'))
    pet = _touch(str(ses / 'pet' / 'sub-1_ses-1_run-01_pet.nii.gz'))
    _write_json(pet.replace('.nii.gz', '.json'), {'TracerRadionuclide': 'F18'})

    session_data = {
        'subjectID': 'sub-1',
        'sessionID': 'ses-1',
        'session_path': str(ses),
        't1_filenames': [t1],
        'flair_filenames': [],
        't2w_filenames': [],
        'dti_filenames': [dwi_lr, dwi_rl],
        'rsf_filenames': [rsf_lr, rsf_rl],
        'nm_filenames': [],
        'perf_filenames': [perf],
        'pet3d_filenames': [pet],
    }

    plan = plan_session_inputs(session_data)
    tracking = plan['selection_tracking']

    assert tracking['t1']['selector'] == 'T1Selector'
    assert tracking['rsf']['selector'] == 'RestingStateSelector'
    assert tracking['dti']['selector'] == 'DwiSelector'
    assert tracking['perf']['selector'] == 'PerfusionSelector'
    assert tracking['pet3d']['selector'] == 'Pet3DSelector'

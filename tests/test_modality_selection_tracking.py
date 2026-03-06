import json
import os

from antsxmm.execution_plan import build_execution_plan
from antsxmm.inputs import plan_session_inputs



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



def test_other_modalities_use_ranked_defaults_and_tracking(tmp_path):
    bids = tmp_path / 'BIDS'
    ses = bids / 'sub-1' / 'ses-1'

    t1a = _touch(str(ses / 'anat' / 'sub-1_ses-1_run-02_T1w.nii.gz'))
    t1b = _touch(str(ses / 'anat' / 'sub-1_ses-1_run-01_T1w.nii.gz'))
    t2w = _touch(str(ses / 'anat' / 'sub-1_ses-1_run-01_T2w.nii.gz'))
    flair = _touch(str(ses / 'anat' / 'sub-1_ses-1_run-02_FLAIR.nii.gz'))

    perf_bad = _touch(str(ses / 'perf' / 'sub-1_ses-1_run-01_m0scan.nii.gz'))
    perf_good = _touch(str(ses / 'perf' / 'sub-1_ses-1_run-02_asl.nii.gz'))

    pet_bad = _touch(str(ses / 'pet' / 'sub-1_ses-1_run-02_petx.nii.gz'))
    pet_good = _touch(str(ses / 'pet' / 'sub-1_ses-1_run-01_pet.nii.gz'))

    nm2 = _touch(str(ses / 'melanin' / 'sub-1_ses-1_run-02_NM.nii.gz'))
    nm1 = _touch(str(ses / 'melanin' / 'sub-1_ses-1_run-01_NM.nii.gz'))

    _write_json(perf_good.replace('.nii.gz', '.json'), {'ArterialSpinLabelingType': 'PCASL'})
    _write_json(pet_good.replace('.nii.gz', '.json'), {'TracerRadionuclide': 'F18'})

    session_data = {
        'subjectID': 'sub-1',
        'sessionID': 'ses-1',
        'session_path': str(ses),
        't1_filenames': [t1a, t1b],
        'flair_filenames': [flair],
        't2w_filenames': [t2w],
        'dti_filenames': [],
        'rsf_filenames': [],
        'nm_filenames': [nm2, nm1],
        'perf_filenames': [perf_bad, perf_good],
        'pet3d_filenames': [pet_bad, pet_good],
    }

    plan = plan_session_inputs(session_data)
    used = plan['used']
    tracking = plan['selection_tracking']

    assert used['t1_filename'] == os.path.realpath(t1b)
    assert used['flair_or_t2_as_flair_filename'] == os.path.realpath(flair)
    assert used['perf_filename'] == os.path.realpath(perf_good)
    assert used['pet3d_filename'] == os.path.realpath(pet_good)
    assert used['nm_filenames'] == [os.path.realpath(nm1), os.path.realpath(nm2)]

    assert tracking['t1']['selected'] == [os.path.realpath(t1b)]
    assert tracking['flair_or_t2']['selected'] == [os.path.realpath(flair)]
    assert tracking['perf']['selected'] == [os.path.realpath(perf_good)]
    assert tracking['pet3d']['selected'] == [os.path.realpath(pet_good)]
    assert tracking['nm']['selected'] == [os.path.realpath(nm1), os.path.realpath(nm2)]

    perf_reasons = tracking['perf']['ranked_candidates'][0]['reasons']
    assert 'exact_suffix:asl' in perf_reasons



def test_execution_plan_uses_ranked_selection_for_other_modalities(tmp_path):
    bids = tmp_path / 'BIDS'
    ses = bids / 'sub-2' / 'ses-1'

    t1a = _touch(str(ses / 'anat' / 'sub-2_ses-1_run-02_T1w.nii.gz'))
    t1b = _touch(str(ses / 'anat' / 'sub-2_ses-1_run-01_T1w.nii.gz'))
    flair = _touch(str(ses / 'anat' / 'sub-2_ses-1_run-01_FLAIR.nii.gz'))
    perf_good = _touch(str(ses / 'perf' / 'sub-2_ses-1_run-01_asl.nii.gz'))
    pet_good = _touch(str(ses / 'pet' / 'sub-2_ses-1_run-01_pet.nii.gz'))

    session_data = {
        'subjectID': 'sub-2',
        'sessionID': 'ses-1',
        'session_path': str(ses),
        't1_filenames': [t1a, t1b],
        'flair_filenames': [flair],
        't2w_filenames': [],
        'dti_filenames': [],
        'rsf_filenames': [],
        'nm_filenames': [],
        'perf_filenames': [perf_good],
        'pet3d_filenames': [pet_good],
    }

    plan = build_execution_plan(session_data, str(tmp_path / 'out'), 'proj')
    by_modality = {u.modality: u for u in plan}

    assert by_modality['T1w'].input_paths == (os.path.realpath(t1b),)
    assert by_modality['T2Flair'].input_paths == (os.path.realpath(flair),)
    assert by_modality['perf'].input_paths == (os.path.realpath(perf_good),)
    assert by_modality['pet3d'].input_paths == (os.path.realpath(pet_good),)

from pathlib import Path

from antsxmm.inputs import _select_rsf_filenames, plan_session_inputs


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__('json').dumps(payload), encoding='utf-8')
    return str(path)


def test_select_rsf_prefers_exact_bold_complementary_pair(tmp_path):
    func = tmp_path / 'sub-1' / 'ses-1' / 'func'
    lr_bold = _touch(func / 'sub-1_ses-1_task-rest_run-01_dir-LR_bold.nii.gz')
    _write_json(func / 'sub-1_ses-1_task-rest_run-01_dir-LR_bold.json', {'PhaseEncodingDirection': 'i'})
    lr_bolda = _touch(func / 'sub-1_ses-1_task-rest_run-01_dir-LR_bolda.nii.gz')
    _write_json(func / 'sub-1_ses-1_task-rest_run-01_dir-LR_bolda.json', {'PhaseEncodingDirection': 'i'})
    rl_bold = _touch(func / 'sub-1_ses-1_task-rest_run-01_dir-RL_bold.nii.gz')
    _write_json(func / 'sub-1_ses-1_task-rest_run-01_dir-RL_bold.json', {'PhaseEncodingDirection': 'i-'})
    rl_bolda = _touch(func / 'sub-1_ses-1_task-rest_run-01_dir-RL_bolda.nii.gz')
    _write_json(func / 'sub-1_ses-1_task-rest_run-01_dir-RL_bolda.json', {'PhaseEncodingDirection': 'i-'})

    selected = _select_rsf_filenames([lr_bolda, lr_bold, rl_bolda, rl_bold])

    assert selected == [lr_bold, rl_bold]


def test_select_rsf_uses_json_phase_direction_when_filename_is_ambiguous(tmp_path):
    func = tmp_path / 'sub-2' / 'ses-1' / 'func'
    rest_a = _touch(func / 'sub-2_ses-1_task-rest_run-01_bold.nii.gz')
    _write_json(func / 'sub-2_ses-1_task-rest_run-01_bold.json', {'PhaseEncodingDirection': 'i'})
    rest_b = _touch(func / 'sub-2_ses-1_task-rest_run-02_bold.nii.gz')
    _write_json(func / 'sub-2_ses-1_task-rest_run-02_bold.json', {'PhaseEncodingDirection': 'i-'})
    rest_c = _touch(func / 'sub-2_ses-1_task-rest_run-03_bold.nii.gz')
    _write_json(func / 'sub-2_ses-1_task-rest_run-03_bold.json', {'PhaseEncodingDirection': 'j'})

    selected = _select_rsf_filenames([rest_c, rest_b, rest_a])

    assert selected == [rest_a, rest_b]


def test_plan_session_inputs_applies_ranked_rsf_selection(tmp_path):
    ses = tmp_path / 'sub-3' / 'ses-1'
    anat = ses / 'anat'
    func = ses / 'func'

    t1 = _touch(anat / 'sub-3_ses-1_T1w.nii.gz')
    lr_bolda = _touch(func / 'sub-3_ses-1_task-rest_run-01_dir-LR_bolda.nii.gz')
    _write_json(func / 'sub-3_ses-1_task-rest_run-01_dir-LR_bolda.json', {'PhaseEncodingDirection': 'i'})
    rl_bold = _touch(func / 'sub-3_ses-1_task-rest_run-01_dir-RL_bold.nii.gz')
    _write_json(func / 'sub-3_ses-1_task-rest_run-01_dir-RL_bold.json', {'PhaseEncodingDirection': 'i-'})
    lr_bold = _touch(func / 'sub-3_ses-1_task-rest_run-01_dir-LR_bold.nii.gz')
    _write_json(func / 'sub-3_ses-1_task-rest_run-01_dir-LR_bold.json', {'PhaseEncodingDirection': 'i'})

    plan = plan_session_inputs({
        'subjectID': 'sub-3',
        'sessionID': 'ses-1',
        'session_path': str(ses),
        't1_filenames': [t1],
        'flair_filenames': [],
        't2w_filenames': [],
        'dti_filenames': [],
        'rsf_filenames': [lr_bolda, rl_bold, lr_bold],
        'nm_filenames': [],
        'perf_filenames': [],
        'pet3d_filenames': [],
    })

    assert plan['rsf_selected_raw'] == [lr_bold, rl_bold]
    assert plan['used']['rsf_filenames'] == [str(Path(lr_bold).resolve()), str(Path(rl_bold).resolve())]

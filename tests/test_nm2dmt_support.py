from antsxmm.execution_plan import build_execution_plan, modality_from_path


def test_modality_mapping_nm_suffix():
    assert modality_from_path('sub-1_ses-1_run-01_NM.nii.gz') == 'NM2DMT'


def test_build_execution_plan_includes_nm2dmt(tmp_path):
    base = tmp_path / 'sub-1' / 'ses-1'
    (base / 'anat').mkdir(parents=True)
    (base / 'melanin').mkdir(parents=True)
    t1 = str(base / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    nm = str(base / 'melanin' / 'sub-1_ses-1_run-001_NM.nii.gz')
    for p in (t1, nm):
        open(p, 'wb').close()

    plan = build_execution_plan(
        {
            'subjectID': 'sub-1',
            'sessionID': 'ses-1',
            't1_filenames': [t1],
            'nm_filenames': [nm],
        },
        output_root=str(tmp_path / 'out'),
        project_id='Proj',
    )

    modalities = [u.modality for u in plan]
    assert 'NM2DMT' in modalities
    nm_unit = next(u for u in plan if u.modality == 'NM2DMT')
    assert nm_unit.output_prefix.endswith('Proj/sub-1/ses-1/NM2DMT/run-01/Proj+sub-1+ses-1+NM2DMT+run-01')

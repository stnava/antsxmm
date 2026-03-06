from antsxmm.execution_plan import build_execution_plan, validate_execution_plan


def test_build_execution_plan_emits_canonical_units(tmp_path):
    base = tmp_path / 'sub-1' / 'ses-1'
    (base / 'anat').mkdir(parents=True)
    (base / 'perf').mkdir(parents=True)
    (base / 'func').mkdir(parents=True)
    t1 = str(base / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    asl = str(base / 'perf' / 'sub-1_ses-1_run-001_asl.nii.gz')
    bold = str(base / 'func' / 'sub-1_ses-1_task-rest_run-001_bold.nii.gz')
    for p in (t1, asl, bold):
        open(p, 'wb').close()

    plan = build_execution_plan(
        {
            'subjectID': 'sub-1',
            'sessionID': 'ses-1',
            't1_filenames': [t1],
            'perf_filenames': [asl],
            'rsf_filenames': [bold],
        },
        output_root=str(tmp_path / 'out'),
        project_id='Proj',
    )

    assert [u.modality for u in plan] == ['T1w', 'perf', 'rsfMRI', 'T1wHierarchical']
    assert all(u.run == 'run-01' for u in plan)
    assert plan[1].output_prefix.endswith('Proj/sub-1/ses-1/perf/run-01/Proj+sub-1+ses-1+perf+run-01')


def test_validate_execution_plan_rejects_duplicate_modality_output(tmp_path):
    from antsxmm.execution_plan import ExecutionUnit

    prefix = str(tmp_path / 'out' / 'Proj+sub-1+ses-1+perf+run-01')
    unit_a = ExecutionUnit('Proj', 'sub-1', 'ses-1', 'perf', 'run-01', ('a.nii.gz',), prefix)
    unit_b = ExecutionUnit('Proj', 'sub-1', 'ses-1', 'perf', 'run-01', ('b.nii.gz',), prefix)

    try:
        validate_execution_plan([unit_a, unit_b])
    except ValueError as e:
        assert 'duplicate execution unit' in str(e) or 'duplicate output prefix' in str(e)
    else:
        raise AssertionError('expected duplicate plan validation failure')

from antsxmm.bids_entities import parse_entities


def test_parse_entities_extracts_standard_fields():
    ent = parse_entities('sub-EAS002_ses-01_task-rest_run-001_bold.nii.gz')
    assert ent['sub'] == 'EAS002'
    assert ent['ses'] == '01'
    assert ent['task'] == 'rest'
    assert ent['run'] == '001'
    assert ent['suffix'] == 'bold'

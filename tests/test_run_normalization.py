from antsxmm.run_id import normalize_run_id


def test_normalize_run_from_run_token():
    assert normalize_run_id('run-001') == 'run-01'


def test_normalize_run_from_integerish_string():
    assert normalize_run_id('1') == 'run-01'


def test_normalize_run_missing_defaults_to_run01():
    assert normalize_run_id(None) == 'run-01'

import logging

from antsxmm.environment import apply_default_environment, get_effective_environment_policy, resolve_thread_count


def test_resolve_thread_count_prefers_antsxmm_threads():
    cfg = resolve_thread_count(environ={"ANTSXMM_THREADS": "6", "SLURM_CPUS_PER_TASK": "12"})
    assert cfg.thread_count == 6
    assert cfg.source == "ANTSXMM_THREADS"


def test_resolve_thread_count_uses_slurm_when_present():
    cfg = resolve_thread_count(environ={"SLURM_CPUS_PER_TASK": "10"})
    assert cfg.thread_count == 10
    assert cfg.source == "SLURM_CPUS_PER_TASK"


def test_apply_default_environment_sets_missing_values_and_preserves_existing(caplog):
    env = {
        "SLURM_CPUS_PER_TASK": "10",
        "OPENBLAS_NUM_THREADS": "2",
    }

    with caplog.at_level(logging.INFO):
        result = apply_default_environment(environ=env, logger=logging.getLogger("antsxmm.test"))

    assert result["thread_count"] == 10
    assert result["thread_source"] == "SLURM_CPUS_PER_TASK"
    assert env["TF_NUM_INTEROP_THREADS"] == "10"
    assert env["TF_NUM_INTRAOP_THREADS"] == "10"
    assert env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "10"
    assert env["MKL_NUM_THREADS"] == "10"
    assert env["MPLBACKEND"] == "Agg"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert result["applied"]["TF_NUM_INTEROP_THREADS"] == "10"
    assert result["preserved"]["OPENBLAS_NUM_THREADS"] == "2"
    assert "Environment thread policy: thread_count=10 source=SLURM_CPUS_PER_TASK" in caplog.text
    assert "Environment OPENBLAS_NUM_THREADS=2 (preserved)" in caplog.text
    assert "Environment TF_NUM_INTEROP_THREADS=10 (defaulted by antsxmm)" in caplog.text


def test_apply_default_environment_falls_back_to_default_thread_count():
    env = {}
    result = apply_default_environment(default_thread_count=7, environ=env)
    assert result["thread_count"] == 7
    assert env["TF_NUM_INTEROP_THREADS"] == "7"
    assert env["OPENBLAS_NUM_THREADS"] == "7"
    assert env["MPLBACKEND"] == "Agg"


def test_get_effective_environment_policy_reports_defaults_without_mutating():
    env = {}
    result = get_effective_environment_policy(default_thread_count=5, environ=env)
    assert result["thread_count"] == 5
    assert result["thread_source"] == "default"
    assert result["effective"]["TF_NUM_INTEROP_THREADS"] == "5"
    assert "TF_NUM_INTEROP_THREADS" not in env

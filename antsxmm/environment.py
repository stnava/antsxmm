from __future__ import annotations

import logging
import os
from dataclasses import dataclass


THREAD_ENV_KEYS = (
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

NON_THREAD_ENV_DEFAULTS = {
    "MPLBACKEND": "Agg",
}


@dataclass(frozen=True)
class EnvironmentConfig:
    thread_count: int
    source: str


def _coerce_positive_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolve_thread_count(default_thread_count: int = 8, environ: dict[str, str] | None = None) -> EnvironmentConfig:
    env = os.environ if environ is None else environ

    explicit = _coerce_positive_int(env.get("ANTSXMM_THREADS"))
    if explicit is not None:
        return EnvironmentConfig(thread_count=explicit, source="ANTSXMM_THREADS")

    slurm = _coerce_positive_int(env.get("SLURM_CPUS_PER_TASK"))
    if slurm is not None:
        return EnvironmentConfig(thread_count=slurm, source="SLURM_CPUS_PER_TASK")

    return EnvironmentConfig(thread_count=int(default_thread_count), source="default")


def _default_values(default_thread_count: int = 8, environ: dict[str, str] | None = None) -> tuple[EnvironmentConfig, dict[str, str]]:
    env = os.environ if environ is None else environ
    cfg = resolve_thread_count(default_thread_count=default_thread_count, environ=env)
    defaults = {key: str(cfg.thread_count) for key in THREAD_ENV_KEYS}
    defaults.update(NON_THREAD_ENV_DEFAULTS)
    return cfg, defaults


def get_effective_environment_policy(
    default_thread_count: int = 8,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    """Return the effective environment policy without mutating the environment."""
    env = os.environ if environ is None else environ
    cfg, defaults = _default_values(default_thread_count=default_thread_count, environ=env)

    effective: dict[str, str] = {}
    for key, default_value in defaults.items():
        current = env.get(key)
        effective[key] = str(current) if current is not None and str(current).strip() != "" else default_value

    return {
        "thread_count": cfg.thread_count,
        "thread_source": cfg.source,
        "effective": effective,
    }


def format_environment_policy_for_log(policy: dict[str, object]) -> list[str]:
    """Format a stable human-readable environment summary for logging."""
    lines = [
        f"Environment thread policy: thread_count={policy['thread_count']} source={policy['thread_source']}"
    ]
    effective = policy.get("effective", {})
    if isinstance(effective, dict):
        for key in (*THREAD_ENV_KEYS, *NON_THREAD_ENV_DEFAULTS.keys()):
            if key in effective:
                lines.append(f"Environment {key}={effective[key]} (effective)")
    return lines


def apply_default_environment(
    default_thread_count: int = 8,
    environ: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Apply stable process-wide runtime defaults.

    Defaults are only applied when the corresponding environment variable is
    absent or empty, so scheduler/user/container overrides remain authoritative.
    Child processes inherit these values automatically.
    """
    env = os.environ if environ is None else environ
    cfg, defaults = _default_values(default_thread_count=default_thread_count, environ=env)

    applied: dict[str, str] = {}
    preserved: dict[str, str] = {}

    for key, value in defaults.items():
        current = env.get(key)
        if current is None or str(current).strip() == "":
            env[key] = value
            applied[key] = value
        else:
            preserved[key] = str(current)

    payload: dict[str, object] = {
        "thread_count": cfg.thread_count,
        "thread_source": cfg.source,
        "effective": {key: env.get(key, "") for key in defaults},
        "applied": applied,
        "preserved": preserved,
    }

    if logger is not None:
        logger.info(
            "Environment thread policy: thread_count=%s source=%s",
            payload["thread_count"],
            payload["thread_source"],
        )
        for key in defaults:
            if key in applied:
                logger.info("Environment %s=%s (defaulted by antsxmm)", key, env[key])
            else:
                logger.info("Environment %s=%s (preserved)", key, env[key])

    return payload

from __future__ import annotations

from .environment import apply_default_environment


def main() -> int | None:
    """Bootstrap antsxmm with early environment policy application.

    This is the preferred console entry point. It applies stable process-wide
    defaults before importing the heavier pipeline module so thread-sensitive
    libraries observe the intended policy during import/initialization.
    """
    apply_default_environment()

    try:
        from .pipeline import entry_point
    except ImportError:  # pragma: no cover - fallback for local execution
        from antsxmm.pipeline import entry_point

    return entry_point()

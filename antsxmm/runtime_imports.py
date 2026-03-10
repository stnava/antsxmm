from __future__ import annotations

import contextlib
import importlib
import io
import types
import warnings
from typing import Any


def import_optional_module(module_name: str) -> Any:
    """Import an optional runtime dependency without CLI-noise side effects.

    Some scientific dependencies emit warnings or print informational messages at
    import time. That behavior is hostile to a CLI, especially for `--help` and
    lightweight commands. We suppress stdout/stderr and known warning chatter
    during import while still returning a usable module object.
    """
    sink = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                return importlib.import_module(module_name)
    except Exception:
        return types.SimpleNamespace()

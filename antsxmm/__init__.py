try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("antsxmm")
    except PackageNotFoundError:
        __version__ = "0.0.0-unknown"

from .bids import parse_antsxbids_layout
from .core import process_session
from .pipeline import run_study

__all__ = ['parse_antsxbids_layout', 'process_session', 'run_study', '__version__']

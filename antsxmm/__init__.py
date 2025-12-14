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
from .core import bind_mm_rows
from .core import check_modality_order
from .core import build_wide_table_from_mmwide

__all__ = ['parse_antsxbids_layout', 'build_wide_table_from_mmwide', 'bind_mm_rows',  'process_session', 'run_study', '__version__']

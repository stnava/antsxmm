"""ANTsXMM publication-grade visual reporting and component library."""

from __future__ import annotations

from antsxmm import __version__
from antsxmm.visualize.core import (
    figure_to_base64,
    image_to_base64,
    render_carpet_plot,
    render_correlation_matrix,
    render_ortho_montage,
    render_slice_gallery,
    render_violin_plot,
)
from antsxmm.visualize.report import (
    generate_session_report,
    generate_study_report,
)
from antsxmm.visualize.theme import (
    build_html_document,
    get_theme_css,
    get_theme_js,
    render_badge,
    render_card,
    render_kpi_card,
    render_tabs,
)

__all__ = [
    "__version__",
    "build_html_document",
    "get_theme_css",
    "get_theme_js",
    "render_badge",
    "render_card",
    "render_kpi_card",
    "render_tabs",
    "figure_to_base64",
    "image_to_base64",
    "render_ortho_montage",
    "render_slice_gallery",
    "render_carpet_plot",
    "render_violin_plot",
    "render_correlation_matrix",
    "generate_session_report",
    "generate_study_report",
]

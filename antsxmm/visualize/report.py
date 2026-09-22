"""Session and Study Multi-Modality Aggregator and Report Generation.

Provides publication-grade, interactive, offline HTML report generation for:
1. Individual subject/session multimodal acquisitions (`generate_session_report`).
2. Cohort-wide study aggregated datasets (`generate_study_report`).

All generated HTML documents are 100% self-contained with zero external CDN dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from antsxmm.visualize.core import figure_to_base64  # noqa: E402
from antsxmm.visualize.modalities import (  # noqa: E402
    AVAILABLE_MODALITIES,
    MODALITY_ALIASES,
    ModalityReport,
    coalesce_multi_row_df,
    find_file,
    find_run_dir,
    safe_read_csv,
    visualize_modality,
)
from antsxmm.visualize.theme import (  # noqa: E402
    build_html_document,
    render_badge,
    render_card,
    render_kpi_card,
    render_tabs,
)

logger = logging.getLogger(__name__)

# Dark theme palette constants matching core.py
DARK_BG = "#0b0f19"
DARK_PANEL_BG = "#162032"
DARK_TEXT = "#f8fafc"
DARK_MUTED = "#94a3b8"
DARK_BORDER = "#27354a"
ACCENT_CYAN = "#38bdf8"
ACCENT_GREEN = "#34d399"
ACCENT_RED = "#f87171"
ACCENT_AMBER = "#f59e0b"

# Modality clinical metadata
MODALITY_METADATA: dict[str, dict[str, str]] = {
    "T1w": {
        "title": "T1w Structural MRI",
        "desc": "High-Resolution Anatomical Brain Extraction & 6-Tissue Segmentation",
        "category": "Structural",
    },
    "T1wHierarchical": {
        "title": "T1w Hierarchical Segmentation",
        "desc": "Deep Subcortical Nuclei, CIT168, Brainstem & DKT Cortical Parcellation",
        "category": "Structural",
    },
    "T2Flair": {
        "title": "T2 FLAIR WMH Lesions",
        "desc": "White Matter Hyperintensity Segmentation & Lesion Burden Quantification",
        "category": "Structural",
    },
    "DTI": {
        "title": "Diffusion Tensor Imaging (DTI)",
        "desc": "White Matter Microstructure (FA, MD) & 3D Gradient Sampling Quality",
        "category": "Diffusion",
    },
    "rsfMRI": {
        "title": "Resting-State fMRI (rsfMRI)",
        "desc": "BOLD Timeseries Carpet Plot, Motion Profile & 22-Network Functional Connectivity",
        "category": "Functional",
    },
    "perf": {
        "title": "Arterial Spin Labeling (ASL / CBF)",
        "desc": "Quantitative Cerebral Blood Flow (CBF) & M0 Perfusion Coregistration",
        "category": "Perfusion",
    },
    "pet3d": {
        "title": "3D Positron Emission Tomography",
        "desc": "Tracer Uptake Distribution & Regional DKT Standardized Uptake Values (SUVR)",
        "category": "Molecular",
    },
    "NM2DMT": {
        "title": "Neuromelanin MRI (NM-2D-MT)",
        "desc": "High-Resolution Midbrain Slab Average & Substantia Nigra Contrast Ratio (SNCR)",
        "category": "Neuromelanin",
    },
}

# Modality column prefix mappings for study tables
MODALITY_PREFIX_MAP: dict[str, list[str]] = {
    "T1w": ["T1w+", "T1+", "t1w_"],
    "T1wHierarchical": ["T1Hier+", "T1wHierarchical+", "hierarchical_"],
    "T2Flair": ["T2Flair+", "wmh_", "FLAIR+"],
    "DTI": ["DTI+", "dti_"],
    "rsfMRI": ["rsfMRI+", "fmri_", "bold_"],
    "perf": ["perf+", "asl_", "cbf_"],
    "pet3d": ["pet3d+", "pet_"],
    "NM2DMT": ["NM2DMT+", "nm_"],
}


@dataclass
class SessionMetadata:
    """Discovered session and subject metadata."""

    subject_id: str
    session_id: str
    project_id: str
    created_utc: str | None = None
    execution_engine: str | None = None
    success: bool | None = None
    tool_version: str | None = None
    input_files_count: int = 0
    status_file_path: Path | None = None
    diagnostics_path: Path | None = None
    study_csv_path: Path | None = None


def _discover_session_metadata(session_dir: Path) -> SessionMetadata:
    """Inspect session directory to extract subject, session, and project metadata."""
    sub_id: str | None = None
    ses_id: str | None = None
    proj_id: str | None = None
    created_utc: str | None = None
    exec_engine: str | None = None
    success: bool | None = None
    tool_ver: str | None = None
    input_files_count = 0

    # 1. Inspect .antsxmm_status.json
    status_file = session_dir / ".antsxmm_status.json"
    if status_file.is_file():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                sdata = json.load(f)
            sub_id = sdata.get("subjectID") or sdata.get("subject_id")
            ses_id = sdata.get("sessionID") or sdata.get("session_id")
            proj_id = sdata.get("project_id") or sdata.get("projectID")
            created_utc = sdata.get("created_utc")
            exec_engine = sdata.get("execution_engine")
            success = sdata.get("success")
            if isinstance(sdata.get("args"), dict):
                tool_ver = sdata["args"].get("tool_version")
            input_files_count = len(sdata.get("input_fingerprint", {}).get("files", []))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", status_file, exc)

    # 2. Inspect study.csv in session directory
    study_csv = find_file(session_dir, "*study.csv")
    if study_csv is not None:
        try:
            df = pd.read_csv(study_csv)
            if len(df) > 0:
                first = df.iloc[0]
                if not proj_id and "projectID" in first and pd.notna(first["projectID"]):
                    proj_id = str(first["projectID"])
                if not sub_id and "subjectID" in first and pd.notna(first["subjectID"]):
                    sub_id = str(first["subjectID"])
                if not ses_id and "date" in first and pd.notna(first["date"]):
                    ses_id = str(first["date"])
        except Exception as exc:
            logger.warning("Failed to inspect %s: %s", study_csv, exc)

    # 3. Inspect input diagnostics
    diag_file = find_file(session_dir, "*input_diagnostics.json")

    # 4. Fallback inference from directory hierarchy
    if not ses_id:
        ses_id = session_dir.name
    if not sub_id:
        sub_id = session_dir.parent.name
    if not proj_id:
        proj_id = session_dir.parent.parent.name

    return SessionMetadata(
        subject_id=sub_id or "Unknown-Subject",
        session_id=ses_id or "Unknown-Session",
        project_id=proj_id or "Unknown-Project",
        created_utc=created_utc,
        execution_engine=exec_engine or "antsxmm",
        success=success,
        tool_version=tool_ver,
        input_files_count=input_files_count,
        status_file_path=status_file if status_file.is_file() else None,
        diagnostics_path=diag_file,
        study_csv_path=study_csv,
    )


def _discover_modalities(session_dir: Path) -> dict[str, Path | None]:
    """Identify present and missing canonical modalities within session_dir."""
    modality_dirs: dict[str, Path | None] = {}

    for mod in AVAILABLE_MODALITIES:
        target_dir = session_dir / mod
        if target_dir.is_dir():
            modality_dirs[mod] = target_dir
            continue

        # Check aliases
        found = False
        for alias, canon in MODALITY_ALIASES.items():
            if canon == mod:
                candidate = session_dir / alias
                if candidate.is_dir():
                    modality_dirs[mod] = candidate
                    found = True
                    break
        if not found:
            modality_dirs[mod] = None

    return modality_dirs


def _extract_executive_kpis(
    reports: dict[str, ModalityReport],
    session_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Extract top-level cross-modality KPIs for Executive Overview banner."""
    kpis: dict[str, dict[str, Any]] = {
        "brain_volume": {
            "label": "Brain Volume",
            "value": "N/A",
            "unit": "cm³",
            "status": "neutral",
            "tooltip": "Total intracranial / parenchymal brain volume",
        },
        "mean_cbf": {
            "label": "Mean CBF",
            "value": "N/A",
            "unit": "ml/100g/min",
            "status": "neutral",
            "tooltip": "Global mean cerebral blood flow (ASL perfusion)",
        },
        "mean_fa": {
            "label": "Mean FA",
            "value": "N/A",
            "unit": "",
            "status": "neutral",
            "tooltip": "Mean fractional anisotropy (DTI white matter microstructure)",
        },
        "mean_pet": {
            "label": "Mean PET SUVR",
            "value": "N/A",
            "unit": "SUVR",
            "status": "neutral",
            "tooltip": "Whole-brain mean tracer uptake (3D PET)",
        },
        "mean_fd": {
            "label": "Mean Framewise Disp",
            "value": "N/A",
            "unit": "mm",
            "status": "neutral",
            "tooltip": "Mean head motion per frame during resting-state fMRI",
        },
    }

    # 1. Brain Volume: check T1wHierarchical, then T1w
    struct_rep = reports.get("T1wHierarchical") or reports.get("T1w")
    if struct_rep:
        for k in struct_rep.kpis:
            if "Total Brain Volume" in k.get("label", "") or "Brain Volume" in k.get("label", ""):
                kpis["brain_volume"]["value"] = k.get("value", "N/A")
                kpis["brain_volume"]["unit"] = k.get("unit", "cm³")
                kpis["brain_volume"]["status"] = k.get("status", "normal")
                break
            elif "ICV" in k.get("label", "") and kpis["brain_volume"]["value"] == "N/A":
                kpis["brain_volume"]["value"] = k.get("value", "N/A")
                kpis["brain_volume"]["unit"] = k.get("unit", "cm³")
                kpis["brain_volume"]["status"] = k.get("status", "normal")

    # If still N/A, check tissues CSV directly
    if kpis["brain_volume"]["value"] == "N/A":
        tissues_csv = find_file(session_dir, "*tissues.csv")
        if tissues_csv:
            tdf = safe_read_csv(tissues_csv)
            if tdf is not None and not tdf.empty:
                vol_cols = [c for c in tdf.columns if "vol_" in c.lower() and "csf" not in c.lower()]
                if vol_cols:
                    tot_vox = tdf[vol_cols].iloc[0].sum()
                    kpis["brain_volume"]["value"] = f"{tot_vox / 1000.0:.1f}"
                    kpis["brain_volume"]["unit"] = "cm³"
                    kpis["brain_volume"]["status"] = "normal"

    # 2. Mean CBF from perf
    perf_rep = reports.get("perf")
    if perf_rep:
        for k in perf_rep.kpis:
            if "CBF" in k.get("label", "") or "cbf" in k.get("label", "").lower():
                kpis["mean_cbf"]["value"] = k.get("value", "N/A")
                kpis["mean_cbf"]["unit"] = k.get("unit", "ml/100g/min")
                kpis["mean_cbf"]["status"] = k.get("status", "normal")
                break

    # 3. Mean FA from DTI
    dti_rep = reports.get("DTI")
    if dti_rep:
        for k in dti_rep.kpis:
            if "Mean FA" in k.get("label", ""):
                kpis["mean_fa"]["value"] = k.get("value", "N/A")
                kpis["mean_fa"]["unit"] = k.get("unit", "")
                kpis["mean_fa"]["status"] = k.get("status", "normal")
                break

    # 4. Mean PET from pet3d
    pet_rep = reports.get("pet3d")
    if pet_rep:
        for k in pet_rep.kpis:
            if "PET" in k.get("label", "") or "Uptake" in k.get("label", ""):
                kpis["mean_pet"]["value"] = k.get("value", "N/A")
                kpis["mean_pet"]["unit"] = k.get("unit", "SUVR")
                kpis["mean_pet"]["status"] = k.get("status", "normal")
                break

    # 5. Mean FD from rsfMRI
    fmri_rep = reports.get("rsfMRI")
    if fmri_rep:
        for k in fmri_rep.kpis:
            if "Mean FD" in k.get("label", "") or k.get("label", "") == "FD Mean":
                val = k.get("value", "N/A")
                kpis["mean_fd"]["value"] = val
                kpis["mean_fd"]["unit"] = k.get("unit", "mm")
                try:
                    num_val = float(val)
                    if num_val > 0.5:
                        kpis["mean_fd"]["status"] = "danger"
                    elif num_val > 0.2:
                        kpis["mean_fd"]["status"] = "warning"
                    else:
                        kpis["mean_fd"]["status"] = "normal"
                except Exception:
                    kpis["mean_fd"]["status"] = k.get("status", "normal")
                break

    return kpis


def _build_session_overview_html(
    meta: SessionMetadata,
    discovered_modalities: dict[str, Path | None],
    reports: dict[str, ModalityReport],
    session_dir: Path,
    group_id: str,
) -> str:
    """Construct the Executive Overview tab HTML content."""
    # Top Cross-Modality KPIs
    top_kpis = _extract_executive_kpis(reports, session_dir)
    kpi_cards_html = "".join(
        render_kpi_card(
            label=item["label"],
            value=item["value"],
            unit=item["unit"],
            status=item["status"],
            tooltip=item["tooltip"],
        )
        for item in top_kpis.values()
    )
    kpis_section = f'<div class="kpi-grid">\n{kpi_cards_html}\n</div>'

    # Metadata Banner
    total_canon = len(AVAILABLE_MODALITIES)
    present_canon = sum(1 for p in discovered_modalities.values() if p is not None)
    success_count = sum(1 for r in reports.values() if r.status in ("success", "partial"))
    warning_count = sum(1 for r in reports.values() if r.status == "warning")
    error_count = sum(1 for r in reports.values() if r.status == "error")

    if error_count > 0:
        overall_badge = render_badge(f"{error_count} Errors", "danger")
    elif warning_count > 0:
        overall_badge = render_badge(f"{warning_count} Warnings", "warning")
    elif present_canon > 0:
        overall_badge = render_badge(f"{present_canon}/{total_canon} Complete", "normal")
    else:
        overall_badge = render_badge("No Modalities", "neutral")

    date_str = meta.created_utc[:19].replace("T", " ") + " UTC" if meta.created_utc else "Recorded on disk"

    meta_banner_html = f"""<div class="card" style="margin-bottom: 20px;">
  <div class="card-header">
    <div class="card-title-group">
      <h3 class="card-title">Subject & Session Executive Overview</h3>
      <span class="card-subtitle">ANTsXMM Automated Multimodal Neuroimaging Pipeline</span>
    </div>
    <div>{overall_badge}</div>
  </div>
  <div class="card-body">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Subject Identifier</div>
        <div style="font-family: var(--font-mono); font-size: 15px; font-weight: 700; color: var(--text-primary); margin-top: 2px;">{html.escape(meta.subject_id)}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Session Identifier</div>
        <div style="font-family: var(--font-mono); font-size: 15px; font-weight: 700; color: var(--text-primary); margin-top: 2px;">{html.escape(meta.session_id)}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Project / Cohort</div>
        <div style="font-size: 15px; font-weight: 600; color: var(--text-primary); margin-top: 2px;">{html.escape(meta.project_id)}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Processing Date</div>
        <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">{html.escape(date_str)}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Pipeline Engine</div>
        <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">{html.escape(meta.execution_engine)} {html.escape(meta.tool_version or '')}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Session Directory</div>
        <div style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); margin-top: 4px; word-break: break-all;">{html.escape(str(session_dir))}</div>
      </div>
    </div>
  </div>
</div>"""

    # Modality Completion Matrix Table
    matrix_rows: list[str] = []
    for mod in AVAILABLE_MODALITIES:
        mod_meta = MODALITY_METADATA.get(mod, {"title": mod, "desc": "", "category": "General"})
        dir_path = discovered_modalities.get(mod)
        is_present = dir_path is not None
        rep = reports.get(mod)

        tab_id = mod.lower()

        if is_present and rep:
            if rep.status == "success":
                status_badge = render_badge("Completed", "normal")
            elif rep.status == "warning":
                status_badge = render_badge("Warning", "warning")
            elif rep.status == "partial":
                status_badge = render_badge("Partial", "info")
            else:
                status_badge = render_badge("Error", "danger")

            # Extract sample KPI label
            kpi_desc = ""
            if rep.kpis:
                first_kpi = rep.kpis[0]
                kpi_desc = f"{first_kpi['label']}: {first_kpi['value']} {first_kpi.get('unit', '')}".strip()
            else:
                kpi_desc = "Pipeline outputs verified"

            btn_html = (
                f'<button type="button" class="theme-toggle-btn" '
                f'style="padding: 4px 10px; font-size: 11px; border-radius: var(--radius-sm);" '
                f'onclick="const b = document.getElementById(\'tab-{group_id}-{tab_id}\'); if (b) b.click();">'
                f'View Tab &rarr;</button>'
            )
        else:
            status_badge = render_badge("Not Acquired", "neutral")
            kpi_desc = '<span style="color: var(--text-muted); font-style: italic;">Omitted in acquisition protocol</span>'
            btn_html = '<span style="color: var(--text-muted); font-size: 12px;">—</span>'

        cat_badge = render_badge(mod_meta["category"], "info")

        matrix_rows.append(f"""<tr>
  <td style="padding: 12px 16px; font-weight: 600; color: var(--text-primary);">
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-family: var(--font-mono); font-size: 13px;">{html.escape(mod)}</span>
      {cat_badge}
    </div>
    <div style="font-size: 12px; font-weight: 400; color: var(--text-muted); margin-top: 2px;">{html.escape(mod_meta["desc"])}</div>
  </td>
  <td style="padding: 12px 16px; text-align: center;">{status_badge}</td>
  <td style="padding: 12px 16px; font-size: 13px; color: var(--text-secondary);">{kpi_desc}</td>
  <td style="padding: 12px 16px; text-align: right;">{btn_html}</td>
</tr>""")

    matrix_table_html = f"""<div style="overflow-x: auto;">
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
    <thead>
      <tr style="border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;">
        <th style="padding: 10px 16px;">Modality & Description</th>
        <th style="padding: 10px 16px; text-align: center;">Status</th>
        <th style="padding: 10px 16px;">Primary Quality Metric</th>
        <th style="padding: 10px 16px; text-align: right;">Action</th>
      </tr>
    </thead>
    <tbody style="divide-y: 1px solid var(--border-color);">
      {"".join(matrix_rows)}
    </tbody>
  </table>
</div>"""

    matrix_card = render_card(
        title="8-Modality Completion Matrix",
        subtitle="Clinical status across all supported neuroimaging modalities",
        content_html=matrix_table_html,
        badge=f"{present_canon} / {total_canon} Present",
    )

    return f"""{meta_banner_html}
<div style="margin-bottom: 8px;">
  <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px; font-weight: 600;">Cross-Modality Key Performance Indicators</h3>
</div>
{kpis_section}
{matrix_card}"""


def generate_session_report(
    session_dir: str | Path,
    output_html: str | Path | None = None,
    theme: str = "auto",
    title: str | None = None,
) -> Path:
    """Generate an offline, standalone interactive HTML report for a subject session.

    Discovers all modalities within `session_dir`, extracts metadata, renders an
    Executive Overview tab with cross-modality KPIs and completion badges, and generates
    specialized diagnostic visualization tabs for each present modality.

    Parameters
    ----------
    session_dir : str or Path
        Directory containing processed session outputs (e.g. pymm/Project/sub-01/ses-01).
    output_html : str or Path, optional
        Target path for generated HTML file. Defaults to `<session_dir>/session_report.html`.
    theme : str, optional
        Color theme: "auto" (default, adaptive), "dark", or "light".
    title : str, optional
        Custom title for report. Defaults to "ANTsXMM Session Report — <sub_id> <ses_id>".

    Returns
    -------
    Path
        Absolute path to the generated HTML report file.
    """
    s_path = Path(session_dir).resolve()
    if not s_path.is_dir():
        raise FileNotFoundError(f"Session directory does not exist: {s_path}")

    # Discover metadata and modalities
    meta = _discover_session_metadata(s_path)
    discovered = _discover_modalities(s_path)

    doc_title = title or f"ANTsXMM Session Report — {meta.subject_id} {meta.session_id}"
    subtitle_text = f"Project: {meta.project_id} &bull; Subject: {meta.subject_id} &bull; Session: {meta.session_id}"

    # Generate individual modality reports for present modalities
    reports: dict[str, ModalityReport] = {}
    for mod, mod_dir in discovered.items():
        if mod_dir is None:
            continue
        try:
            logger.info("Visualizing modality '%s' from %s", mod, mod_dir)
            rep = visualize_modality(mod, mod_dir, session_dir=s_path)
            reports[mod] = rep
        except Exception as exc:
            logger.exception("Failed to visualize modality '%s': %s", mod, exc)
            error_html = render_card(
                title=f"{mod} — Generation Error",
                subtitle="An unexpected error occurred during report generation",
                content_html=f'<div style="color: var(--status-danger-text); font-family: var(--font-mono); font-size: 12px; padding: 12px; background-color: var(--status-danger-bg); border-radius: var(--radius-sm); border: 1px solid var(--status-danger-border);">{html.escape(str(exc))}</div>',
                badge="Error",
            )
            reports[mod] = ModalityReport(
                name=mod,
                title=mod,
                status="error",
                kpis=[],
                html_content=error_html,
                errors=[str(exc)],
            )

    # Build tabs
    group_id = f"session-{uuid.uuid4().hex[:6]}"
    tabs_data: list[tuple[str, str, str]] = []

    # 1. Executive Overview tab
    overview_content = _build_session_overview_html(meta, discovered, reports, s_path, group_id)
    tabs_data.append(("overview", "Executive Overview", overview_content))

    # 2. Present modality tabs
    for mod in AVAILABLE_MODALITIES:
        if mod in reports:
            rep = reports[mod]
            mod_title = MODALITY_METADATA.get(mod, {}).get("title", mod)
            tab_id = mod.lower()
            tabs_data.append((tab_id, mod, rep.html_content))

    tabs_html = render_tabs(tabs_data, active_tab="overview", tab_group_id=group_id)

    # Standalone HTML Document
    full_html = build_html_document(
        title=doc_title,
        body_html=tabs_html,
        theme=theme,
        header_title=doc_title,
        subtitle=subtitle_text,
    )

    # Resolve output path
    if output_html is None:
        out_path = s_path / "session_report.html"
    else:
        out_path = Path(output_html).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_html, encoding="utf-8")
    logger.info("Wrote session report to %s (%d bytes)", out_path, len(full_html))
    return out_path


# ============================================================================
# Study-Level Report Aggregation
# ============================================================================

def _compute_study_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute subject x modality presence/absence matrix from study dataframe."""
    subjects: list[str] = []
    sessions: list[str] = []
    projects: list[str] = []

    for idx, row in df.iterrows():
        subjects.append(str(row.get("subject_id", f"Sub-{idx}")))
        sessions.append(str(row.get("session_id", f"Ses-{idx}")))
        projects.append(str(row.get("project_id", "Study")))

    matrix_dict: dict[str, list[int]] = {mod: [] for mod in AVAILABLE_MODALITIES}

    for _, row in df.iterrows():
        for mod in AVAILABLE_MODALITIES:
            prefixes = MODALITY_PREFIX_MAP.get(mod, [f"{mod}+"])
            matching_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
            if matching_cols:
                # Check if at least one column is non-null
                non_null_count = row[matching_cols].notna().sum()
                matrix_dict[mod].append(1 if non_null_count > 0 else 0)
            else:
                # Check if modality column matches directly
                if "modality" in row and str(row["modality"]).lower() == mod.lower():
                    matrix_dict[mod].append(1)
                else:
                    matrix_dict[mod].append(0)

    matrix_df = pd.DataFrame(matrix_dict)
    matrix_df["Subject"] = subjects
    matrix_df["Session"] = sessions
    matrix_df["Project"] = projects
    return matrix_df


def _render_missingness_heatmap(matrix_df: pd.DataFrame) -> str:
    """Render a publication-grade completeness heatmap figure and encode as Base64."""
    mod_cols = [c for c in AVAILABLE_MODALITIES if c in matrix_df.columns]
    n_subs = len(matrix_df)
    n_mods = len(mod_cols)

    # Calculate dynamic figure height based on subject count
    fig_height = max(3.5, min(12.0, 0.45 * n_subs + 2.0))
    fig_width = max(8.0, min(14.0, 1.2 * n_mods + 3.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        data = matrix_df[mod_cols].values
        labels = [f"{row['Subject']} ({row['Project']})" for _, row in matrix_df.iterrows()]

        # Binary colormap: slate for missing, cyan for present
        cmap = matplotlib.colors.ListedColormap(["#1e293b", "#38bdf8"])
        bounds = [-0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

        # Set ticks
        ax.set_xticks(np.arange(n_mods))
        ax.set_xticklabels(mod_cols, rotation=35, ha="right", color=DARK_TEXT, fontsize=11, fontweight="bold")
        ax.set_yticks(np.arange(n_subs))
        ax.set_yticklabels(labels, color=DARK_TEXT, fontsize=10, fontfamily="monospace")

        # Text annotations in cells
        for i in range(n_subs):
            for j in range(n_mods):
                val = data[i, j]
                txt = "✓" if val == 1 else "✗"
                txt_color = "#0b0f19" if val == 1 else "#64748b"
                ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=12, fontweight="bold")

        # Gridlines between cells
        ax.set_xticks(np.arange(n_mods + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_subs + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=DARK_BORDER, linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", size=0)

        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

        ax.set_title("Modality Acquisition Matrix across Subjects", color=DARK_TEXT, fontsize=13, fontweight=700, pad=14)

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.6, pad=0.03)
        cbar.ax.set_yticklabels(["Missing", "Acquired"], color=DARK_TEXT, fontsize=10)
        cbar.ax.set_facecolor(DARK_PANEL_BG)
        cbar.outline.set_color(DARK_BORDER)

        fig.tight_layout()
        return figure_to_base64(fig)
    finally:
        plt.close(fig)


def _extract_key_study_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard neuroimaging quantitative scalars across subjects for QC analysis."""
    records: list[dict[str, Any]] = []

    tissue_cols = [
        "T1Hier+vol_gmtissues",
        "T1Hier+vol_wmtissues",
        "T1Hier+vol_deepgraytissues",
        "T1Hier+vol_cerebellumtissues",
        "T1Hier+vol_brainstemtissues",
    ]

    for idx, row in df.iterrows():
        sub = str(row.get("subject_id", f"Sub-{idx}"))
        ses = str(row.get("session_id", f"Ses-{idx}"))
        proj = str(row.get("project_id", "Study"))

        # ICV
        icv = np.nan
        for col in ["T1Hier+icv", "icv"]:
            if col in row and pd.notna(row[col]):
                icv = float(row[col]) / 1000.0  # mm3 to cm3
                break

        # Brain Volume
        brain_vol = np.nan
        if "T1Hier+vol_braintissues" in row and pd.notna(row["T1Hier+vol_braintissues"]):
            brain_vol = float(row["T1Hier+vol_braintissues"]) / 1000.0
        else:
            avail = [row[c] for c in tissue_cols if c in row and pd.notna(row[c])]
            if avail:
                brain_vol = float(sum(avail)) / 1000.0

        # Cortical Thickness
        thk = np.nan
        if "T1Hier+thk_gmtissues" in row and pd.notna(row["T1Hier+thk_gmtissues"]):
            thk = float(row["T1Hier+thk_gmtissues"])
        else:
            thk_cols = [c for c in df.columns if c.startswith("T1Hier+thk_") and "cortex" in c and pd.notna(row[c])]
            if thk_cols:
                thk = float(np.mean([row[c] for c in thk_cols]))

        # Mean FA
        fa = np.nan
        for col in ["DTI+FA_mean", "FA_mean"]:
            if col in row and pd.notna(row[col]):
                fa = float(row[col])
                break

        # Mean CBF
        cbf = np.nan
        for col in ["perf+cbf_mean", "cbf_mean"]:
            if col in row and pd.notna(row[col]):
                cbf = float(row[col])
                break
        if np.isnan(cbf):
            cbf_cols = [c for c in df.columns if c.startswith("perf+cbf_mean_") and pd.notna(row[c])]
            if cbf_cols:
                cbf = float(np.mean([row[c] for c in cbf_cols]))

        # Mean PET
        pet = np.nan
        for col in ["pet3d+brainmask_mean", "pet3d+gm_mean", "pet3d_mean"]:
            if col in row and pd.notna(row[col]):
                pet = float(row[col])
                break
        if np.isnan(pet):
            pet_cols = [c for c in df.columns if c.startswith("pet3d+mean_") and pd.notna(row[c])]
            if pet_cols:
                pet = float(np.mean([row[c] for c in pet_cols]))

        # Mean FD
        fd = np.nan
        for col in ["rsfMRI+fcnxprounset_FD_mean", "FD_mean"]:
            if col in row and pd.notna(row[col]):
                fd = float(row[col])
                break

        # WMH Mass
        wmh = np.nan
        for col in ["T2Flair+wmh_mass", "wmh_mass"]:
            if col in row and pd.notna(row[col]):
                wmh = float(row[col])
                break

        records.append({
            "Subject": sub,
            "Session": ses,
            "Project": proj,
            "ICV (cm³)": icv,
            "Brain Volume (cm³)": brain_vol,
            "Cortical Thickness (mm)": thk,
            "Mean FA": fa,
            "Mean CBF (ml/100g/min)": cbf,
            "Mean PET (SUVR)": pet,
            "Framewise Disp (mm)": fd,
            "WMH Mass (mm³)": wmh,
        })

    return pd.DataFrame(records)


def _render_population_qc_plots(scalars_df: pd.DataFrame) -> str:
    """Generate multi-panel population distribution QC plots (violins, strips, thresholds)."""
    metrics = [
        ("ICV (cm³)", "Intracranial Volume (ICV)", None),
        ("Brain Volume (cm³)", "Total Brain Parenchyma Volume", None),
        ("Cortical Thickness (mm)", "Mean Cortical Thickness", (1.5, 3.5)),
        ("Mean FA", "Diffusion Fractional Anisotropy", (0.2, 0.6)),
        ("Framewise Disp (mm)", "Framewise Displacement (Motion)", (0.0, 0.25)),
        ("Mean CBF (ml/100g/min)", "Cerebral Blood Flow (CBF)", (30.0, 90.0)),
    ]

    # Filter to metrics with at least one non-null value
    active_metrics = [m for m in metrics if m[0] in scalars_df.columns and scalars_df[m[0]].notna().sum() > 0]
    if not active_metrics:
        return ""

    n_panels = len(active_metrics)
    ncols = 3 if n_panels >= 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), dpi=150, facecolor=DARK_BG)
    try:
        axes_flat = np.atleast_1d(axes).flatten()

        for idx, (col_name, title, norm_range) in enumerate(active_metrics):
            ax = axes_flat[idx]
            ax.set_facecolor(DARK_PANEL_BG)

            sub_df = scalars_df.dropna(subset=[col_name]).copy()
            vals = sub_df[col_name].values

            # Normative shading if available
            if norm_range is not None:
                ax.axhspan(norm_range[0], norm_range[1], color=ACCENT_GREEN, alpha=0.12, label="Reference Range")

            # FD specific motion threshold line
            if "Framewise" in title:
                ax.axhline(0.25, color=ACCENT_RED, linestyle="--", linewidth=1.2, label="Motion Cutoff (0.25 mm)")

            # Plot distribution: if N >= 4, use violin or boxplot + strip, else strip with mean bar
            if len(vals) >= 4:
                sns.violinplot(
                    y=col_name,
                    data=sub_df,
                    ax=ax,
                    color=ACCENT_CYAN,
                    inner=None,
                    alpha=0.3,
                    cut=0,
                )

            # Strip plot with jitter
            has_hue = "Project" in sub_df.columns and sub_df["Project"].nunique() > 1
            if has_hue:
                n_proj = sub_df["Project"].nunique()
                palette = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, "#e879f9"][:n_proj]
                sns.stripplot(
                    x=["Cohort"] * len(sub_df),
                    y=col_name,
                    hue="Project",
                    data=sub_df,
                    ax=ax,
                    palette=palette,
                    size=8,
                    jitter=0.2,
                    edgecolor=DARK_BG,
                    linewidth=1,
                )
            else:
                sns.stripplot(
                    x=["Cohort"] * len(sub_df),
                    y=col_name,
                    data=sub_df,
                    ax=ax,
                    color=ACCENT_CYAN,
                    size=8,
                    jitter=0.2,
                    edgecolor=DARK_BG,
                    linewidth=1,
                )

            # Draw mean line
            if len(vals) > 0:
                mean_val = np.mean(vals)
                ax.axhline(mean_val, color=ACCENT_CYAN, linestyle=":", linewidth=1.5, alpha=0.7, label=f"Mean: {mean_val:.2f}")

            ax.set_title(title, color=DARK_TEXT, fontsize=11, fontweight=700, pad=8)
            ax.set_xlabel("")
            ax.set_ylabel(col_name, color=DARK_MUTED, fontsize=10)
            ax.tick_params(colors=DARK_TEXT, labelsize=9)
            ax.grid(True, linestyle=":", alpha=0.3, color=DARK_BORDER, axis="y")

            for spine in ax.spines.values():
                spine.set_color(DARK_BORDER)

            # Legend styling
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles=handles[:3],
                    labels=labels[:3],
                    loc="best",
                    fontsize=8,
                    facecolor=DARK_PANEL_BG,
                    edgecolor=DARK_BORDER,
                    labelcolor=DARK_TEXT,
                )

        # Hide any unused subplots
        for idx in range(len(active_metrics), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        return figure_to_base64(fig)
    finally:
        plt.close(fig)


def _detect_study_outliers(scalars_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify extreme values or high-motion QC outlier flags across cohort."""
    outliers: list[dict[str, Any]] = []

    # 1. Motion outlier check: FD > 0.25 mm
    if "Framewise Disp (mm)" in scalars_df.columns:
        for _, row in scalars_df.iterrows():
            fd = row["Framewise Disp (mm)"]
            if pd.notna(fd) and fd > 0.25:
                severity = "danger" if fd > 0.5 else "warning"
                outliers.append({
                    "subject": row["Subject"],
                    "session": row["Session"],
                    "project": row["Project"],
                    "modality": "rsfMRI",
                    "metric": "Framewise Displacement",
                    "value": f"{fd:.3f} mm",
                    "cutoff": "> 0.25 mm",
                    "reason": "Elevated head motion during fMRI acquisition",
                    "severity": severity,
                })

    # 2. Diffusion anisotropy check: FA < 0.05 or extreme deviation
    if "Mean FA" in scalars_df.columns:
        for _, row in scalars_df.iterrows():
            fa = row["Mean FA"]
            if pd.notna(fa) and fa < 0.02:
                outliers.append({
                    "subject": row["Subject"],
                    "session": row["Session"],
                    "project": row["Project"],
                    "modality": "DTI",
                    "metric": "Fractional Anisotropy",
                    "value": f"{fa:.4f}",
                    "cutoff": "< 0.02",
                    "reason": "Abnormally low whole-brain FA (potential artifact or mask mismatch)",
                    "severity": "warning",
                })

    # 3. Statistical Z-score check if N >= 4
    numeric_cols = [c for c in scalars_df.columns if c not in ("Subject", "Session", "Project")]
    if len(scalars_df) >= 4:
        for col in numeric_cols:
            vals = scalars_df[col].dropna()
            if len(vals) >= 4:
                mean = vals.mean()
                std = vals.std()
                if std > 0:
                    for _, row in scalars_df.iterrows():
                        val = row[col]
                        if pd.notna(val):
                            z = abs((val - mean) / std)
                            if z >= 3.0:
                                outliers.append({
                                    "subject": row["Subject"],
                                    "session": row["Session"],
                                    "project": row["Project"],
                                    "modality": col.split()[0],
                                    "metric": col,
                                    "value": f"{val:.2f}",
                                    "cutoff": f"|z| = {z:.1f} (> 3.0)",
                                    "reason": f"Extreme population deviation (>3 standard deviations from mean {mean:.2f})",
                                    "severity": "danger",
                                })
                            elif z >= 2.0:
                                outliers.append({
                                    "subject": row["Subject"],
                                    "session": row["Session"],
                                    "project": row["Project"],
                                    "modality": col.split()[0],
                                    "metric": col,
                                    "value": f"{val:.2f}",
                                    "cutoff": f"|z| = {z:.1f} (> 2.0)",
                                    "reason": f"Moderate population deviation (>2 standard deviations from mean {mean:.2f})",
                                    "severity": "warning",
                                })

    return outliers


def generate_study_report(
    study_csv_path: str | Path,
    output_html: str | Path | None = None,
    theme: str = "auto",
    title: str | None = None,
) -> Path:
    """Generate an offline, standalone interactive HTML report for a study-level dataset.

    Reads study-level aggregated CSV, computes cohort-wide summary statistics,
    generates a modality completion/missingness matrix heatmap, creates population
    distribution QC plots across subjects, and identifies outlier QC flags.

    Parameters
    ----------
    study_csv_path : str or Path
        Path to aggregated study CSV (e.g. study_multimodal_aggregated.csv).
    output_html : str or Path, optional
        Target path for generated HTML file. Defaults to `<study_dir>/study_report.html`.
    theme : str, optional
        Color theme: "auto" (default, adaptive), "dark", or "light".
    title : str, optional
        Custom title for report. Defaults to "ANTsXMM Study Aggregation & QC Report".

    Returns
    -------
    Path
        Absolute path to the generated HTML report file.
    """
    c_path = Path(study_csv_path).resolve()
    if not c_path.is_file():
        raise FileNotFoundError(f"Study CSV file does not exist: {c_path}")

    logger.info("Generating study report from: %s", c_path)
    df = pd.read_csv(c_path)

    doc_title = title or "ANTsXMM Study Aggregation & Multimodal QC Report"
    sub_title = f"Source: {c_path.name} &bull; Cohort Evaluation Summary"

    # Cohort summary statistics
    n_rows = len(df)
    n_cols = len(df.columns)
    unique_subs = df["subject_id"].nunique() if "subject_id" in df.columns else n_rows
    unique_ses = df["session_id"].nunique() if "session_id" in df.columns else n_rows
    unique_projs = list(df["project_id"].unique()) if "project_id" in df.columns else []

    # Completeness matrix & heatmap
    matrix_df = _compute_study_completeness(df)
    heatmap_b64 = _render_missingness_heatmap(matrix_df)

    # Modality coverage summary
    mod_coverage_rows: list[str] = []
    total_mods_present = 0
    for mod in AVAILABLE_MODALITIES:
        if mod in matrix_df.columns:
            cnt = int(matrix_df[mod].sum())
            pct = (cnt / n_rows * 100.0) if n_rows > 0 else 0.0
            if cnt > 0:
                total_mods_present += 1
            badge = render_badge(f"{pct:.0f}% ({cnt}/{n_rows})", "normal" if pct >= 80 else ("warning" if pct > 0 else "neutral"))
            mod_meta = MODALITY_METADATA.get(mod, {"title": mod, "desc": "", "category": "General"})
            mod_coverage_rows.append(f"""<tr>
  <td style="padding: 10px 16px; font-weight: 600; color: var(--text-primary);">
    <span style="font-family: var(--font-mono);">{html.escape(mod)}</span>
    <span style="font-size: 12px; color: var(--text-muted); font-weight: 400; margin-left: 8px;">{html.escape(mod_meta["title"])}</span>
  </td>
  <td style="padding: 10px 16px; text-align: center;">{badge}</td>
  <td style="padding: 10px 16px; font-size: 13px; color: var(--text-secondary);">{cnt} of {n_rows} sessions available</td>
</tr>""")

    coverage_table_html = f"""<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
  <thead>
    <tr style="border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;">
      <th style="padding: 10px 16px;">Modality</th>
      <th style="padding: 10px 16px; text-align: center;">Cohort Coverage</th>
      <th style="padding: 10px 16px;">Availability</th>
    </tr>
  </thead>
  <tbody style="divide-y: 1px solid var(--border-color);">
    {"".join(mod_coverage_rows)}
  </tbody>
</table>"""

    # Top KPI cards
    kpi_cards = [
        render_kpi_card("Total Sessions", n_rows, status="normal", tooltip="Total processed entity sessions in aggregated dataset"),
        render_kpi_card("Unique Subjects", unique_subs, status="normal", tooltip="Unique subject identifiers"),
        render_kpi_card("Modalities Found", f"{total_mods_present} / {len(AVAILABLE_MODALITIES)}", status="normal" if total_mods_present >= 6 else "warning", tooltip="Number of canonical modalities represented"),
        render_kpi_card("Total Features", f"{n_cols:,}", status="info", tooltip="Total quantitative columns aggregated"),
    ]
    top_kpis_html = f'<div class="kpi-grid">\n{"".join(kpi_cards)}\n</div>'

    # Population QC Plots
    scalars_df = _extract_key_study_scalars(df)
    qc_plots_b64 = _render_population_qc_plots(scalars_df)

    qc_plots_card = ""
    if qc_plots_b64:
        qc_plots_card = render_card(
            title="Population Quantitative Distributions & QC Cutoffs",
            subtitle="Violin, strip, and threshold distributions for key cross-modality neuroimaging scalars",
            content_html=f'<div class="img-container"><img class="img-responsive" src="{qc_plots_b64}" alt="Population QC Distributions"></div>',
            badge="Population QC",
        )

    # Missingness Card
    missingness_card = render_card(
        title="Modality Missingness & Acquisition Matrix",
        subtitle="Subject-by-modality completion status across study cohort",
        content_html=f"""<div class="grid-2" style="align-items: start;">
  <div class="img-container"><img class="img-responsive" src="{heatmap_b64}" alt="Missingness Heatmap"></div>
  <div style="background-color: var(--bg-card); border-radius: var(--radius-md); border: 1px solid var(--border-color); padding: 12px;">{coverage_table_html}</div>
</div>""",
        badge="Completeness",
    )

    # Outlier Detection Table
    outliers = _detect_study_outliers(scalars_df)
    if outliers:
        outlier_rows: list[str] = []
        for o in outliers:
            sev_badge = render_badge(o["severity"].upper(), o["severity"])
            outlier_rows.append(f"""<tr>
  <td style="padding: 10px 14px; font-family: var(--font-mono); font-weight: 600; color: var(--text-primary);">{html.escape(str(o["subject"]))}</td>
  <td style="padding: 10px 14px; font-family: var(--font-mono); color: var(--text-muted);">{html.escape(str(o["session"]))}</td>
  <td style="padding: 10px 14px; font-weight: 500;">{html.escape(str(o["project"]))}</td>
  <td style="padding: 10px 14px; font-weight: 600; color: var(--text-primary);">{html.escape(str(o["metric"]))}</td>
  <td style="padding: 10px 14px; font-family: var(--font-mono); font-weight: 700;">{html.escape(str(o["value"]))}</td>
  <td style="padding: 10px 14px; color: var(--text-muted); font-size: 12px;">{html.escape(str(o["cutoff"]))}</td>
  <td style="padding: 10px 14px; font-size: 12px; color: var(--text-secondary);">{html.escape(str(o["reason"]))}</td>
  <td style="padding: 10px 14px; text-align: center;">{sev_badge}</td>
</tr>""")

        outlier_table_html = f"""<div style="overflow-x: auto;">
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
    <thead>
      <tr style="border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;">
        <th style="padding: 10px 14px;">Subject</th>
        <th style="padding: 10px 14px;">Session</th>
        <th style="padding: 10px 14px;">Cohort</th>
        <th style="padding: 10px 14px;">Metric</th>
        <th style="padding: 10px 14px;">Observed Value</th>
        <th style="padding: 10px 14px;">Cutoff / Reference</th>
        <th style="padding: 10px 14px;">Clinical QC Note</th>
        <th style="padding: 10px 14px; text-align: center;">Alert</th>
      </tr>
    </thead>
    <tbody style="divide-y: 1px solid var(--border-color);">
      {"".join(outlier_rows)}
    </tbody>
  </table>
</div>"""
        outlier_badge = f"{len(outliers)} Alerts"
        outlier_card = render_card(
            title="Quality Control Outliers & Clinical Motion Alerts",
            subtitle="Subjects exceeding physiological thresholds (>2-3 SD or high motion)",
            content_html=outlier_table_html,
            badge=outlier_badge,
        )
    else:
        outlier_card = render_card(
            title="Quality Control Outliers & Clinical Motion Alerts",
            subtitle="Automated statistical screening across all quantitative scalars",
            content_html='<div style="padding: 16px; text-align: center; color: var(--status-normal-text); font-weight: 600;">✓ No statistical outliers or excessive motion alerts detected across this cohort.</div>',
            badge="Clean",
        )

    # Subject Data Table
    sub_table_rows: list[str] = []
    for _, row in scalars_df.iterrows():
        sub_table_rows.append(f"""<tr>
  <td style="padding: 8px 12px; font-family: var(--font-mono); font-weight: 600;">{html.escape(str(row["Subject"]))}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono); color: var(--text-muted);">{html.escape(str(row["Session"]))}</td>
  <td style="padding: 8px 12px;">{html.escape(str(row["Project"]))}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["ICV (cm³)"]:.1f}' if pd.notna(row["ICV (cm³)"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Brain Volume (cm³)"]:.1f}' if pd.notna(row["Brain Volume (cm³)"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Cortical Thickness (mm)"]:.2f}' if pd.notna(row["Cortical Thickness (mm)"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Mean FA"]:.3f}' if pd.notna(row["Mean FA"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Mean CBF (ml/100g/min)"]:.1f}' if pd.notna(row["Mean CBF (ml/100g/min)"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Mean PET (SUVR)"]:.2f}' if pd.notna(row["Mean PET (SUVR)"]) else '—'}</td>
  <td style="padding: 8px 12px; font-family: var(--font-mono);">{f'{row["Framewise Disp (mm)"]:.3f}' if pd.notna(row["Framewise Disp (mm)"]) else '—'}</td>
</tr>""")

    subjects_table_card = render_card(
        title="Cohort Key Quantitative Metrics Summary",
        subtitle="Extracted scalar values for all evaluated subjects and sessions",
        content_html=f"""<div style="overflow-x: auto;">
  <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
    <thead>
      <tr style="border-bottom: 2px solid var(--border-color); color: var(--text-muted); font-size: 11px; text-transform: uppercase;">
        <th style="padding: 8px 12px;">Subject</th>
        <th style="padding: 8px 12px;">Session</th>
        <th style="padding: 8px 12px;">Cohort</th>
        <th style="padding: 8px 12px;">ICV</th>
        <th style="padding: 8px 12px;">Brain Vol</th>
        <th style="padding: 8px 12px;">Cortical Thk</th>
        <th style="padding: 8px 12px;">Mean FA</th>
        <th style="padding: 8px 12px;">Mean CBF</th>
        <th style="padding: 8px 12px;">Mean PET</th>
        <th style="padding: 8px 12px;">Mean FD</th>
      </tr>
    </thead>
    <tbody style="divide-y: 1px solid var(--border-color);">
      {"".join(sub_table_rows)}
    </tbody>
  </table>
</div>""",
        badge=f"{len(scalars_df)} Sessions",
    )

    # Cohort Overview Banner
    projs_str = ", ".join(unique_projs) if unique_projs else "Single Study"
    study_banner_html = f"""<div class="card" style="margin-bottom: 20px;">
  <div class="card-header">
    <div class="card-title-group">
      <h3 class="card-title">Study Aggregated Multi-Modality Overview</h3>
      <span class="card-subtitle">Cohorts: {html.escape(projs_str)}</span>
    </div>
    <div>{render_badge(f"{n_rows} Sessions", "normal")}</div>
  </div>
  <div class="card-body">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Evaluated Cohorts</div>
        <div style="font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 2px;">{html.escape(projs_str)}</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Total Entities</div>
        <div style="font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 2px;">{n_rows} rows</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Features Tracked</div>
        <div style="font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 2px;">{n_cols:,} variables</div>
      </div>
      <div>
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Study Table Path</div>
        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); margin-top: 2px; word-break: break-all;">{html.escape(str(c_path))}</div>
      </div>
    </div>
  </div>
</div>"""

    # Assemble Document
    body_html = f"""{study_banner_html}
<div style="margin-bottom: 8px;">
  <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px; font-weight: 600;">Study-Level Summary Metrics</h3>
</div>
{top_kpis_html}
{missingness_card}
{qc_plots_card}
{outlier_card}
{subjects_table_card}"""

    full_html = build_html_document(
        title=doc_title,
        body_html=body_html,
        theme=theme,
        header_title=doc_title,
        subtitle=sub_title,
    )

    if output_html is None:
        out_path = c_path.parent / "study_report.html"
    else:
        out_path = Path(output_html).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_html, encoding="utf-8")
    logger.info("Wrote study report to %s (%d bytes)", out_path, len(full_html))
    return out_path

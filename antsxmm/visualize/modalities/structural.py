"""Structural T1w and T1wHierarchical modality visualizer.

Provides comprehensive visual reporting for anatomical T1-weighted MRI:
- Skull-stripped brain extraction mask contour on anatomical underlay
- 3-class / multi-class tissue segmentation contours (CSF, GM, WM, Deep, Cerebellum)
- DKT cortical and subcortical parcellation atlas overlays
- Deep structure segmentations (CIT168 subcortical nuclei, brainstem, cerebellum)
- Cortical parcel volume/thickness distribution violin plots
- Volumetric comparison bar charts and clinical KPI metric cards (ICV, Brain Volume, GM/WM ratio)
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from antsxmm.visualize.core import (
    DARK_BG,
    DARK_BORDER,
    DARK_MUTED,
    DARK_PANEL_BG,
    DARK_TEXT,
    figure_to_base64,
    render_ortho_montage,
    render_slice_gallery,
    render_violin_plot,
)
from antsxmm.visualize.modalities import (
    ModalityReport,
    find_file,
    find_run_dir,
    load_nifti_data,
    safe_read_csv,
)
from antsxmm.visualize.theme import render_card, render_kpi_card

logger = logging.getLogger(__name__)


def can_visualize(modality_dir: Path | str) -> bool:
    """Determine if directory contains T1w or T1wHierarchical outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("t1", "structural", "hierarchical")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "brain_n4_dnz.nii.gz",
            "head.nii.gz",
            "tissue_segmentation.nii.gz",
            "dkt_parcellation.nii.gz",
            "*T1w*mmwide.csv",
        )
    )


def _render_volumetric_bars(tissue_df: pd.DataFrame, title: str = "Tissue Compartment Volumes") -> str:
    """Render a horizontal bar chart of brain tissue volumes (cm³)."""
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=150, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        # Map labels to tissue names if present
        label_names = {
            1: "CSF",
            2: "Cortical GM",
            3: "White Matter",
            4: "Deep Gray",
            5: "Brainstem",
            6: "Cerebellum",
        }

        labels = []
        volumes_cm3 = []
        colors = ["#38bdf8", "#34d399", "#fbbf24", "#a78bfa", "#f472b6", "#fb923c"]

        vol_col = "VolumeInMillimeters" if "VolumeInMillimeters" in tissue_df.columns else "VolumeInVoxels"

        for idx, row in tissue_df.iterrows():
            lab = int(row.get("Label", idx + 1))
            name = label_names.get(lab, f"Class {lab}")
            val = float(row[vol_col]) / 1000.0  # mm3 to cm3
            labels.append(name)
            volumes_cm3.append(val)

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, volumes_cm3, color=colors[: len(labels)], edgecolor="none", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=DARK_TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Volume (cm³)", color=DARK_TEXT, fontsize=10)
        ax.tick_params(colors=DARK_MUTED, labelsize=9)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3, color=DARK_BORDER)

        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

        # Bar value annotations
        max_val = max(volumes_cm3) if volumes_cm3 else 1.0
        for bar, val in zip(bars, volumes_cm3):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f} cm³",
                va="center",
                ha="left",
                color=DARK_TEXT,
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlim(0, max_val * 1.25)
        ax.set_title(title, color=DARK_TEXT, fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=150, close=True)
    finally:
        plt.close(fig)


def visualize_structural(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate comprehensive visual report for T1w & T1wHierarchical.

    Parameters
    ----------
    modality_dir : Path or str
        Path to T1w or T1wHierarchical modality folder or run folder.
    session_dir : Path or str, optional
        Parent session directory to locate companion structural folders.
    """
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    # Resolve active run directory
    run_dir = find_run_dir(mod_path)

    # If modality_dir is T1w without images, attempt to find T1wHierarchical
    hier_run_dir = run_dir
    if session_dir is not None:
        sess_p = Path(session_dir)
        hier_cand = sess_p / "T1wHierarchical"
        if hier_cand.is_dir():
            hier_run_dir = find_run_dir(hier_cand)
    elif "hierarchical" not in mod_path.name.lower() and mod_path.parent.is_dir():
        hier_cand = mod_path.parent / "T1wHierarchical"
        if hier_cand.is_dir():
            hier_run_dir = find_run_dir(hier_cand)

    # 1. Locate images and masks
    underlay_file = (
        find_file(hier_run_dir, "brain_n4_dnz.nii.gz")
        or find_file(hier_run_dir, "head.nii.gz")
        or find_file(run_dir, "brain_n4_dnz.nii.gz")
        or find_file(run_dir, "head.nii.gz")
    )
    mask_file = find_file(hier_run_dir, "brain_extraction.nii.gz") or find_file(run_dir, "brain_extraction.nii.gz")
    tiss_file = find_file(hier_run_dir, "tissue_segmentation.nii.gz") or find_file(run_dir, "tissue_segmentation.nii.gz")
    dkt_file = (
        find_file(hier_run_dir, "dkt_parcellation.nii.gz")
        or find_file(hier_run_dir, "dkt_lobes.nii.gz")
        or find_file(hier_run_dir, "dkt_cortex.nii.gz")
    )
    deep_file = (
        find_file(hier_run_dir, "cit168lab.nii.gz")
        or find_file(hier_run_dir, "brainstem.nii.gz")
        or find_file(hier_run_dir, "cerebellum.nii.gz")
    )

    # 2. Locate CSV metrics
    tissues_csv = find_file(hier_run_dir, "tissues.csv") or find_file(run_dir, "tissues.csv")
    icv_csv = find_file(hier_run_dir, "icv.csv") or find_file(run_dir, "icv.csv")
    dktcortex_csv = find_file(hier_run_dir, "dktcortex.csv") or find_file(run_dir, "dktcortex.csv")
    mmwide_csv = (
        find_file(hier_run_dir, "mmwide.csv")
        or find_file(run_dir, "mmwide.csv")
        or find_file(mod_path, "mmwide.csv")
    )

    df_tissues = safe_read_csv(tissues_csv)
    df_icv = safe_read_csv(icv_csv)
    df_dktcortex = safe_read_csv(dktcortex_csv)
    df_mmwide = safe_read_csv(mmwide_csv)

    # 3. Extract key metrics for KPI cards
    icv_val: float | None = None
    if df_icv is not None and not df_icv.empty and "icv" in df_icv.columns:
        icv_val = float(df_icv.iloc[0]["icv"]) / 1000.0  # cm3
    elif df_mmwide is not None and not df_mmwide.empty and "icv" in df_mmwide.columns:
        icv_val = float(df_mmwide.iloc[0]["icv"]) / 1000.0

    brain_vol_cm3: float | None = None
    gm_vol_cm3: float | None = None
    wm_vol_cm3: float | None = None
    gm_wm_ratio: float | None = None

    if df_tissues is not None and not df_tissues.empty:
        vol_col = "VolumeInMillimeters" if "VolumeInMillimeters" in df_tissues.columns else "VolumeInVoxels"
        t_dict = {}
        for _, row in df_tissues.iterrows():
            t_dict[int(row.get("Label", 0))] = float(row[vol_col]) / 1000.0

        gm_vol_cm3 = t_dict.get(2, 0.0)
        wm_vol_cm3 = t_dict.get(3, 0.0)
        deep_gray = t_dict.get(4, 0.0)
        brainstem = t_dict.get(5, 0.0)
        cerebellum = t_dict.get(6, 0.0)
        brain_vol_cm3 = gm_vol_cm3 + wm_vol_cm3 + deep_gray + brainstem + cerebellum

        if wm_vol_cm3 > 0:
            gm_wm_ratio = (gm_vol_cm3 + deep_gray) / wm_vol_cm3

    # Add KPI cards
    if icv_val is not None:
        kpis.append({
            "label": "Intracranial Vol (ICV)",
            "value": f"{icv_val:,.1f}",
            "unit": "cm³",
            "status": "normal",
            "tooltip": "Estimated total intracranial volume derived from skull-stripping template registration.",
        })
    if brain_vol_cm3 is not None and brain_vol_cm3 > 0:
        kpis.append({
            "label": "Total Brain Volume",
            "value": f"{brain_vol_cm3:,.1f}",
            "unit": "cm³",
            "status": "normal",
            "tooltip": "Aggregate parenchymal volume (Cortical GM, WM, Deep Nuclei, Brainstem, Cerebellum).",
        })
    if gm_vol_cm3 is not None and gm_vol_cm3 > 0:
        kpis.append({
            "label": "Cortical Gray Matter",
            "value": f"{gm_vol_cm3:,.1f}",
            "unit": "cm³",
            "status": "normal",
            "tooltip": "Segmented cerebral cortical ribbon volume.",
        })
    if wm_vol_cm3 is not None and wm_vol_cm3 > 0:
        kpis.append({
            "label": "Cerebral White Matter",
            "value": f"{wm_vol_cm3:,.1f}",
            "unit": "cm³",
            "status": "normal",
            "tooltip": "Segmented deep and subcortical white matter volume.",
        })
    if gm_wm_ratio is not None:
        kpis.append({
            "label": "GM / WM Ratio",
            "value": f"{gm_wm_ratio:.2f}",
            "unit": "",
            "status": "normal" if (0.6 <= gm_wm_ratio <= 1.4) else "warning",
            "tooltip": "Ratio of total gray matter to white matter volume (normative range: 0.7 - 1.2).",
        })

    # Render KPI Cards HTML Grid
    kpi_cards_html = "".join(
        render_kpi_card(
            label=k["label"],
            value=k["value"],
            unit=k.get("unit", ""),
            status=k.get("status", "normal"),
            tooltip=k.get("tooltip", ""),
        )
        for k in kpis
    )
    if kpi_cards_html:
        cards_html.append(f'<div class="kpi-grid" style="margin-bottom: 24px;">{kpi_cards_html}</div>')

    # Load 3D arrays
    underlay_arr = load_nifti_data(underlay_file)
    mask_arr = load_nifti_data(mask_file)
    tiss_arr = load_nifti_data(tiss_file)
    dkt_arr = load_nifti_data(dkt_file)
    deep_arr = load_nifti_data(deep_file)

    if underlay_arr is None:
        errors.append(f"No anatomical T1 underlay image found in {hier_run_dir}")
        cards_html.append(
            render_card(
                title="T1w Anatomical Underlay Missing",
                content_html="<p class='text-muted'>Anatomical underlay image (brain_n4_dnz.nii.gz or head.nii.gz) was not found.</p>",
                badge="Missing",
            )
        )
    else:
        # Card 1: Brain Extraction Ortho Montage
        try:
            ortho_uri = render_ortho_montage(
                underlay=underlay_arr,
                overlay=mask_arr,
                overlay_cmap="cool",
                overlay_alpha=0.35,
                title="T1w Brain Extraction & Alignment (Axial / Coronal / Sagittal)",
            )
            img_html = f'<div class="img-container"><img src="{ortho_uri}" class="img-responsive" alt="T1w Ortho Montage" /></div>'
            cards_html.append(
                render_card(
                    title="Brain Extraction & Orthogonal Alignment",
                    subtitle="Skull-stripped brain anatomical underlay with extraction mask boundary overlay",
                    content_html=img_html,
                    badge="Verified",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render ortho montage: {exc}")

        # Card 2: Tissue Segmentation Gallery
        if tiss_arr is not None:
            try:
                gallery_uri = render_slice_gallery(
                    underlay=underlay_arr,
                    overlay=tiss_arr,
                    axis=2,
                    nslices=7,
                    contours=True,
                    overlay_cmap="tab10",
                    title="Multi-Class Tissue Segmentation (1: CSF, 2: GM, 3: WM, 4: Deep, 5: Stem, 6: Cereb)",
                )
                img_html = f'<div class="img-container"><img src="{gallery_uri}" class="img-responsive" alt="Tissue Segmentation" /></div>'
                cards_html.append(
                    render_card(
                        title="Tissue Segmentation Contours",
                        subtitle="Multi-slice axial gallery demonstrating CSF, Gray Matter, and White Matter boundaries",
                        content_html=img_html,
                        badge="6-Class",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render tissue gallery: {exc}")

        # Card 3: DKT Cortical Parcellation Atlas
        if dkt_arr is not None:
            try:
                dkt_gallery_uri = render_slice_gallery(
                    underlay=underlay_arr,
                    overlay=dkt_arr,
                    axis=2,
                    nslices=7,
                    contours=False,
                    overlay_cmap="nipy_spectral",
                    overlay_alpha=0.45,
                    title="Desikan-Killiany-Tourville (DKT) Atlas Parcellation",
                )
                img_html = f'<div class="img-container"><img src="{dkt_gallery_uri}" class="img-responsive" alt="DKT Atlas" /></div>'
                cards_html.append(
                    render_card(
                        title="DKT Cortical & Subcortical Parcellation",
                        subtitle="Atlas parcel labeling overlay aligned with anatomical coordinate space",
                        content_html=img_html,
                        badge="Atlas",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render DKT gallery: {exc}")

        # Card 4: Subcortical & Deep Nuclei Structures
        if deep_arr is not None:
            try:
                deep_uri = render_slice_gallery(
                    underlay=underlay_arr,
                    overlay=deep_arr,
                    axis=1,  # Coronal slice view is ideal for subcortical / brainstem
                    nslices=7,
                    contours=True,
                    overlay_cmap="Set1",
                    title="Subcortical & Deep Nuclei Segmentations (Coronal View)",
                )
                img_html = f'<div class="img-container"><img src="{deep_uri}" class="img-responsive" alt="Deep Structures" /></div>'
                cards_html.append(
                    render_card(
                        title="Deep Nuclei & Subcortical Parcellation",
                        subtitle="High-resolution subcortical nuclei (CIT168, Brainstem, Cerebellar subregions)",
                        content_html=img_html,
                        badge="Subcortical",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render deep structures gallery: {exc}")

    # Card 5: Quantitative Distributions & Volumetric Comparisons
    quant_items = []
    if df_tissues is not None and not df_tissues.empty:
        try:
            bar_uri = _render_volumetric_bars(df_tissues, title="Tissue Compartment Volumetrics (cm³)")
            quant_items.append(f'<div class="img-container" style="flex: 1; min-width: 320px;"><img src="{bar_uri}" class="img-responsive" alt="Tissue Volumes" /></div>')
        except Exception as exc:
            errors.append(f"Failed to render tissue bar chart: {exc}")

    if df_dktcortex is not None and not df_dktcortex.empty:
        try:
            plot_df = df_dktcortex.copy()
            desc_col = "Description" if "Description" in plot_df.columns else plot_df.columns[1]
            plot_df["Hemisphere"] = plot_df[desc_col].apply(
                lambda s: "Left" if str(s).lower().startswith("left") else ("Right" if str(s).lower().startswith("right") else "Bilateral")
            )
            val_col = "VolumeInMillimeters" if "VolumeInMillimeters" in plot_df.columns else "VolumeInVoxels"
            plot_df["Volume_cm3"] = plot_df[val_col] / 1000.0

            violin_uri = render_violin_plot(
                plot_df,
                x="Hemisphere",
                y="Volume_cm3",
                title="Cortical Parcel Volume Distribution by Hemisphere",
            )
            quant_items.append(f'<div class="img-container" style="flex: 1; min-width: 320px;"><img src="{violin_uri}" class="img-responsive" alt="Cortical Distributions" /></div>')
        except Exception as exc:
            errors.append(f"Failed to render cortical violin plot: {exc}")

    if quant_items:
        row_html = f'<div style="display: flex; gap: 16px; flex-wrap: wrap;">{"".join(quant_items)}</div>'
        cards_html.append(
            render_card(
                title="Volumetric Distributions & Parcel Comparisons",
                subtitle="Quantitative compartment volumes and cortical parcel distribution summaries",
                content_html=row_html,
                badge="Quantitative",
            )
        )

    status = "error" if (underlay_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="T1wHierarchical" if "hierarchical" in mod_path.name.lower() else "T1w",
        title="Structural MRI (T1w & Hierarchical Anatomy)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


# Standard alias
visualize_modality = visualize_structural

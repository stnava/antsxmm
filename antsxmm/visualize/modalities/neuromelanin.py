"""NM2DMT Neuromelanin-sensitive MRI modality visualizer.

Provides publication-grade visual reporting for Neuromelanin-sensitive MRI:
- Denoised high-resolution cropped slab average ortho montage
- Substantia Nigra (SN) and Locus Coeruleus (LC) target ROI contour overlays
- Full acquisition slab with midbrain field-of-view bounding box gallery
- T1-to-NM anatomical coregistration boundary check
- Contrast ratio metric summaries and clinical KPI cards (SNCR, LCCR, SNR, reference region).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from antsxmm.visualize.core import render_ortho_montage, render_slice_gallery
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
    """Determine if directory contains NM2DMT neuromelanin outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("nm2dmt", "nm", "neuromelanin")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "NM_avg.nii.gz",
            "NM_avg_cropped.nii.gz",
            "NM_labels.nii.gz",
            "*NM2DMT*mmwide.csv",
        )
    )


def visualize_neuromelanin(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate visual report for NM2DMT Neuromelanin-sensitive MRI."""
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    run_dir = find_run_dir(mod_path)

    # 1. Locate files
    cropped_file = find_file(run_dir, "NM_avg_cropped.nii.gz") or find_file(run_dir, "*cropped*.nii.gz")
    avg_file = find_file(run_dir, "NM_avg.nii.gz") or find_file(run_dir, "*NM_avg*.nii.gz")
    labels_file = find_file(run_dir, "NM_labels.nii.gz") or find_file(run_dir, "*labels*.nii.gz")
    midbrain_file = find_file(run_dir, "NM_midbrainROI.nii.gz") or find_file(run_dir, "*midbrain*.nii.gz")
    t1_nm_file = find_file(run_dir, "t1_to_NM.nii.gz") or find_file(run_dir, "*t1_to*.nii.gz")
    mmwide_csv = find_file(run_dir, "mmwide.csv") or find_file(mod_path, "mmwide.csv")

    df_mmwide = safe_read_csv(mmwide_csv)

    # 2. Extract metrics
    sn_intensity: float | None = None
    ref_intensity: float | None = None
    sncr_val: float | None = None
    lccr_val: float | None = None
    nm_snr: float | None = None
    nm_max: float | None = None

    if df_mmwide is not None and not df_mmwide.empty:
        row = df_mmwide.iloc[0]
        if "NM_avg_refregion" in row and not np.isnan(row["NM_avg_refregion"]):
            ref_intensity = float(row["NM_avg_refregion"])
        elif "NM_ReferenceRegion_Right_Mean" in row and not np.isnan(row["NM_ReferenceRegion_Right_Mean"]):
            ref_intensity = float(row["NM_ReferenceRegion_Right_Mean"])

        if "NM_avg_substantianigra" in row and not np.isnan(row["NM_avg_substantianigra"]):
            sn_intensity = float(row["NM_avg_substantianigra"])

        if "NM_avg_signaltonoise" in row and not np.isnan(row["NM_avg_signaltonoise"]):
            nm_snr = float(row["NM_avg_signaltonoise"])

        if "NM_max" in row and not np.isnan(row["NM_max"]):
            nm_max = float(row["NM_max"])

        # Check for pre-calculated SNCR or LCCR
        if "NM_SNCR" in row and not np.isnan(row["NM_SNCR"]):
            sncr_val = float(row["NM_SNCR"])
        if "NM_LCCR" in row and not np.isnan(row["NM_LCCR"]):
            lccr_val = float(row["NM_LCCR"])

    # 3. Load Images
    cropped_arr = load_nifti_data(cropped_file)
    avg_arr = load_nifti_data(avg_file)
    labels_arr = load_nifti_data(labels_file)
    midbrain_arr = load_nifti_data(midbrain_file)
    t1_nm_arr = load_nifti_data(t1_nm_file)

    underlay_arr = cropped_arr if cropped_arr is not None else avg_arr

    # If SN intensity was 0 or unrecorded in CSV, try calculating directly from voxels
    if (sn_intensity is None or sn_intensity == 0.0) and cropped_arr is not None and labels_arr is not None:
        if labels_arr.shape == cropped_arr.shape and np.any(labels_arr > 0):
            sn_vox = cropped_arr[labels_arr > 0]
            if len(sn_vox) > 0:
                sn_intensity = float(np.mean(sn_vox))

    # Calculate SNCR: Contrast Ratio = (S_SN - S_ref) / S_ref * 100%
    if sncr_val is None and sn_intensity is not None and ref_intensity is not None and ref_intensity > 0:
        if sn_intensity > 0:
            sncr_val = ((sn_intensity - ref_intensity) / ref_intensity) * 100.0

    # 4. Build KPI Metric Cards
    if sncr_val is not None:
        kpis.append({
            "label": "SN Contrast Ratio (SNCR)",
            "value": f"{sncr_val:+.1f}%",
            "unit": "",
            "status": "normal" if sncr_val > 5.0 else "warning",
            "tooltip": "Substantia Nigra contrast ratio relative to crus cerebri reference region (diagnostic for dopaminergic loss).",
        })
    else:
        kpis.append({
            "label": "SN Contrast Ratio (SNCR)",
            "value": "N/A" if (sn_intensity is None or sn_intensity == 0.0) else f"{sn_intensity:.1f}",
            "unit": "",
            "status": "neutral",
            "tooltip": "Substantia Nigra contrast ratio.",
        })

    if lccr_val is not None:
        kpis.append({
            "label": "LC Contrast Ratio (LCCR)",
            "value": f"{lccr_val:+.1f}%",
            "unit": "",
            "status": "normal",
            "tooltip": "Locus Coeruleus contrast ratio relative to pontine reference parenchyma.",
        })

    if ref_intensity is not None:
        kpis.append({
            "label": "Reference Region Intensity",
            "value": f"{ref_intensity:.1f}",
            "unit": "a.u.",
            "status": "normal",
            "tooltip": "Mean signal intensity in cerebral peduncle / crus cerebri reference region.",
        })

    if nm_snr is not None and nm_snr > 0:
        kpis.append({
            "label": "NM Slab SNR",
            "value": f"{nm_snr:.2f}",
            "unit": "",
            "status": "normal" if nm_snr >= 3.0 else "warning",
            "tooltip": "Signal-to-noise ratio across neuromelanin-sensitive slab volumes.",
        })

    if underlay_arr is not None:
        kpis.append({
            "label": "Slab Slices Acquired",
            "value": f"{underlay_arr.shape[-1]}",
            "unit": "slices",
            "status": "normal",
            "tooltip": "Thick-slice coverage across brainstem substantia nigra and locus coeruleus.",
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

    # Card 1: Cropped Midbrain Slab Ortho Montage
    if underlay_arr is not None:
        try:
            ortho_uri = render_ortho_montage(
                underlay=underlay_arr,
                overlay=labels_arr if (labels_arr is not None and labels_arr.shape == underlay_arr.shape) else None,
                overlay_cmap="autumn",
                overlay_alpha=0.5,
                title="NM2DMT High-Resolution Midbrain Slab (Cropped View)",
            )
            img_html = f'<div class="img-container"><img src="{ortho_uri}" class="img-responsive" alt="NM Slab Montage" /></div>'
            cards_html.append(
                render_card(
                    title="Cropped Midbrain Slab Average & Substantia Nigra Spatial Coverage",
                    subtitle="3-view orthogonal montage zoomed on brainstem with target neuromelanin ROI overlays",
                    content_html=img_html,
                    badge="NM2DMT Slab",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render NM ortho montage: {exc}")

    # Card 2: Target ROI Gallery (SN & LC Slices)
    if cropped_arr is not None:
        try:
            n_slices = min(7, cropped_arr.shape[2])
            gallery_uri = render_slice_gallery(
                underlay=cropped_arr,
                overlay=labels_arr if (labels_arr is not None and labels_arr.shape == cropped_arr.shape) else None,
                axis=2,
                nslices=n_slices,
                contours=True,
                overlay_cmap="autumn",
                title="Target Nuclei ROI Segmentations (Substantia Nigra & Locus Coeruleus)",
            )
            img_html = f'<div class="img-container"><img src="{gallery_uri}" class="img-responsive" alt="Target ROIs" /></div>'
            cards_html.append(
                render_card(
                    title="Target Nuclei ROI Segmentations & Spatial Overlays",
                    subtitle="High-resolution slice gallery displaying localized dopaminergic and noradrenergic target masks",
                    content_html=img_html,
                    badge="Segmentation",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render NM ROI gallery: {exc}")

    # Card 3: Full Acquisition Slab with Midbrain Bounding Box
    if avg_arr is not None and midbrain_arr is not None and avg_arr.shape == midbrain_arr.shape:
        try:
            n_slices = min(7, avg_arr.shape[2])
            mb_uri = render_slice_gallery(
                underlay=avg_arr,
                overlay=midbrain_arr,
                axis=2,
                nslices=n_slices,
                contours=True,
                overlay_cmap="spring",
                title="Full Thick-Slice Acquisition with Midbrain Target Bounding Box",
            )
            img_html = f'<div class="img-container"><img src="{mb_uri}" class="img-responsive" alt="Midbrain Bounding Box" /></div>'
            cards_html.append(
                render_card(
                    title="Full Acquisition Slab & Field-of-View Placement",
                    subtitle="Multi-slice gallery of uncropped neuromelanin sequence with midbrain crop boundary",
                    content_html=img_html,
                    badge="Acquisition QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render full NM slab gallery: {exc}")

    # Card 4: T1-to-NM Coregistration Check
    if cropped_arr is not None and t1_nm_arr is not None and cropped_arr.shape == t1_nm_arr.shape:
        try:
            t1_coreg_uri = render_slice_gallery(
                underlay=cropped_arr,
                overlay=(t1_nm_arr > np.percentile(t1_nm_arr[t1_nm_arr > 0], 30)).astype(float) if np.any(t1_nm_arr > 0) else None,
                axis=2,
                nslices=min(7, cropped_arr.shape[2]),
                contours=True,
                overlay_cmap="cool",
                title="T1 Anatomical Coregistration Alignment Contours",
            )
            img_html = f'<div class="img-container"><img src="{t1_coreg_uri}" class="img-responsive" alt="T1 Coregistration" /></div>'
            cards_html.append(
                render_card(
                    title="T1 Structural Coregistration Verification",
                    subtitle="Axial gallery verifying alignment between resampled high-resolution T1 and neuromelanin slab",
                    content_html=img_html,
                    badge="Coregistration QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render T1-NM coregistration gallery: {exc}")

    status = "error" if (underlay_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="NM2DMT",
        title="Neuromelanin-Sensitive MRI (NM2DMT)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


visualize_modality = visualize_neuromelanin

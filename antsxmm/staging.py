import os
import math
import re
import shutil
import tempfile
import traceback
from pathlib import Path
import types

try:  # optional dependency for lightweight installs / unit tests
    import ants  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ants = types.SimpleNamespace()

def extract_image_id(filename):
    """
    Backwards-compatible helper for extracting an image/run ID from a BIDS-like filename.

    Canonical form is run-XX (e.g. run-01). Defaults to run-01 if missing.
    Also accepts legacy patterns like r0002 and maps them to run-02 when possible.
    """
    fname = os.path.basename(filename)

    m = re.search(r"run-(\d+)", fname)
    if m:
        return f"run-{int(m.group(1)):02d}"

    m = re.search(r"_(?:r)(\d+)[_.]", fname)
    if m:
        # r0002 -> run-02 (best-effort)
        return f"run-{int(m.group(1)):02d}"

    return "run-01"


def get_modality_variant(filename, base_modality, sep):
    """
    Returns the specific antspymm modality string with direction appended.
    """
    fname = os.path.basename(filename)
    suffix = ""
    
    if "dir-RL" in fname or "dir-PA" in fname or "_RL" in fname or "_PA" in fname:
        suffix = "RL"
    elif "dir-LR" in fname or "dir-AP" in fname or "_LR" in fname or "_AP" in fname:
        suffix = "LR"
        
    if suffix:
        if sep == "_":
            return "{}{}".format(base_modality, suffix)
        else:
            return "{}{}{}".format(base_modality, sep, suffix)
    
    # Default mappings
    if base_modality == "dwi": return "DTI"
    if base_modality == "func": return "rsfMRI"
    
    return base_modality


def sanitize_and_stage_file(filepath, project, subject, date, base_modality, image_id, sep, staging_root, verbose=False):
    """
    Stages a file into a strict NRG directory structure in tmp.
    """
    # Pandas rows may contain NaN for missing entries; those come through as
    # floats and will break os.path operations. Treat all missing/non-path
    # values as absent.
    if filepath is None:
        return None, None, None
    if isinstance(filepath, float) and pd.isna(filepath):
        return None, None, None
    if not isinstance(filepath, (str, os.PathLike)):
        return None, None, None

    filepath = os.fspath(filepath)
    if not filepath:
        return None, None, None

    modality = get_modality_variant(filepath, base_modality, sep)
    
    # Clean modality string for filename construction if separator matches
    filename_modality = modality
    if sep in modality:
        filename_modality = modality.replace(sep, "")

    # Ensure extension is handled
    name = os.path.basename(filepath)
    if name.endswith(".nii.gz"):
        ext = ".nii.gz"
    elif name.endswith(".nii"):
        ext = ".nii"
    else:
        ext = os.path.splitext(name)[1]

    # FORCE NRG FILENAME
    safe_sub = subject if subject else "sub"
    safe_date = date if date else "ses"
    
    new_filename = "{}_{}_{}_{}{}".format(safe_sub, safe_date, filename_modality, image_id, ext)
    
    # Construct NRG path
    dest_dir = os.path.join(staging_root, project, subject, date, modality, image_id)
    os.makedirs(dest_dir, exist_ok=True)
    
    symlink_path = os.path.join(dest_dir, new_filename)
    
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
        
    os.symlink(os.path.abspath(filepath), symlink_path)
    
    if verbose:
        print(" Staged: {} -> {}/{}/{}".format(name, modality, image_id, new_filename))
        
    # Stage sidecars (bval, bvec, json)
    new_filename_base = new_filename.replace(ext, "")
    
    if name.endswith(".nii.gz"):
        orig_base = name[:-7]
    elif name.endswith(".nii"):
        orig_base = name[:-4]
    else:
        orig_base = os.path.splitext(name)[0]
        
    src_dir = os.path.dirname(filepath)
    sidecars = [".bval", ".bvec", ".json"]
    
    for side_ext in sidecars:
        src_side = os.path.join(src_dir, orig_base + side_ext)
        if os.path.exists(src_side):
            dst_side = os.path.join(dest_dir, new_filename_base + side_ext)
            if os.path.exists(dst_side) or os.path.islink(dst_side):
                os.remove(dst_side)
            os.symlink(os.path.abspath(src_side), dst_side)
            if verbose:
                print("   + Sidecar: {} -> {}".format(side_ext, os.path.basename(dst_side)))

    return symlink_path, modality, image_id

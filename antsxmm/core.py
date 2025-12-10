import os
import antspymm
import pandas as pd
import ants
import tempfile
import shutil
import re

def sanitize_filename(filepath, requirements, verbose=False):
    """
    Checks if filepath meets antspymm naming requirements.
    If not, creates a symlink in a temporary directory that does.
    
    Args:
        filepath: path to file
        requirements: list of strings (e.g. ['lair']) that MUST be in filename
    
    Returns:
        Path to original file or sanitized symlink.
    """
    if not filepath:
        return None
        
    filename = os.path.basename(filepath)
    
    # Check if any requirement is met exactly
    meets_req = False
    for req in requirements:
        if req in filename:
            meets_req = True
            break
            
    if meets_req:
        return filepath
        
    # Validation failed, creating symlink
    if verbose:
        print(f"Sanitizing filename: {filename} does not contain {requirements}")
        
    # Create temp directory
    staging_dir = os.path.join(tempfile.gettempdir(), "antsxmm_staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    # Construct new name
    injector = requirements[0]
    
    # Attempt case-insensitive replacement (e.g. FLAIR -> flair)
    new_name = filename
    replaced = False
    for req in requirements:
        # Create a regex pattern to match 'req' ignoring case
        pattern = re.compile(re.escape(req), re.IGNORECASE)
        # If we find the pattern (e.g. LAIR inside FLAIR), replace it with the requirement (lair)
        # BUT, to fix the "Flair" vs "flair" issue, if the match is part of a larger word
        # we might want to be careful. 
        # Simpler approach: If "FLAIR" is in the name, replace the WHOLE word "FLAIR" with "flair"
        # if possible.
        
        # Current logic: Replace 'LAIR' with 'lair'. 'FLAIR' -> 'Flair'.
        # 'Flair' contains 'lair', so it passes antspymm check.
        if pattern.search(filename):
            new_name = pattern.sub(req, filename)
            replaced = True
            break
            
    if not replaced:
        # Just insert the requirement before the extension
        name, ext = os.path.splitext(filename)
        if name.endswith(".nii"): # handle .nii.gz
            name, ext2 = os.path.splitext(name)
            ext = ext2 + ext
        new_name = f"{name}_{injector}{ext}"
        
    symlink_path = os.path.join(staging_dir, new_name)
    
    # Remove existing if present
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
        
    os.symlink(os.path.abspath(filepath), symlink_path)
    return symlink_path

def process_session(session_data, output_root, project_id="ANTsX", 
                    denoise_dti=True, dti_moco='SyN', verbose=True):
    """
    Runs the ANTsPyMM pipeline for a single session row.
    """
    
    # 1. Setup paths
    sub_id = session_data['subjectID']
    date_id = session_data['date']
    image_uid = '000' 
    
    # 2. Extract and Sanitize filenames
    t1_fn = session_data['t1_filename']
    
    # Fix FLAIR vs lair issue
    flair_raw = session_data.get('flair_filename', None)
    flair_fn = sanitize_filename(flair_raw, ["lair"], verbose)
    
    # Fix BOLD vs fMRI/func issue
    rsf_raw = session_data.get('rsf_filenames', [])
    rsf_fns = [sanitize_filename(f, ["fMRI", "func"], verbose) for f in rsf_raw]
    
    # DTI and NM usually match ("dwi", "NM"), but sanitizing checks don't hurt
    dti_fns = session_data.get('dti_filenames', [])
    nm_fns = session_data.get('nm_filenames', [])

    # Mock source dir for antspymm requirement
    mock_source_dir = os.path.dirname(os.path.dirname(t1_fn)) 
    
    try:
        if verbose:
            print(f"Generating MM DataFrame for {sub_id} - {date_id}")
            
        study_csv = antspymm.generate_mm_dataframe(
            projectID=project_id,
            subjectID=sub_id,
            date=date_id,
            imageUniqueID=image_uid,
            modality='T1w',
            source_image_directory=mock_source_dir,
            output_image_directory=output_root,
            t1_filename=t1_fn,
            flair_filename=flair_fn,
            rsf_filenames=rsf_fns,
            dti_filenames=dti_fns,
            nm_filenames=nm_fns
        )
        
        # Clean up columns that are None/NaN
        study_csv_clean = study_csv.dropna(axis=1)
        
        # 3. Get Standard Templates
        try:
            template_path = antspymm.get_data("PPMI_template0", target_extension=".nii.gz")
            mask_path = antspymm.get_data("PPMI_template0_brainmask", target_extension=".nii.gz")
            
            if not template_path or not mask_path:
                if verbose: print("Template not found locally.")
                template = None
            else:
                template = ants.image_read(template_path)
                template_mask = ants.image_read(mask_path)
                template = template * template_mask
                template = ants.crop_image(template, ants.iMath(template_mask, "MD", 12))
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load default templates: {e}")
            template = None

        # 4. Run the Pipeline
        if verbose:
            print(f"Starting execution for {sub_id}...")
            
        antspymm.mm_csv(
            study_csv_clean,
            dti_motion_correct=dti_moco,
            dti_denoise=denoise_dti,
            normalization_template=template,
            normalization_template_output='ppmi',
            normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
            normalization_template_spacing=[1,1,1]
        )
        
        return True

    except Exception as e:
        print(f"Error processing {sub_id} {date_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

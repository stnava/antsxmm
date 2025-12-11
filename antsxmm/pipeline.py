import click
import os
import pandas as pd
from tqdm import tqdm
import antspymm
import antspyt1w

# Local imports
try:
    from .bids import parse_antsxbids_layout
    from .core import process_session
    from ._version import version as __version__
except ImportError:
    from antsxmm.bids import parse_antsxbids_layout
    from antsxmm.core import process_session
    try:
        from importlib.metadata import version
        __version__ = version("antsxmm")
    except:
        __version__ = "0.0.0-dev"

def run_study(bids_dir, output_dir, project, denoise_dti=True, participant_label=None, session_label=None):
    print(f"Parsing BIDS layout from: {bids_dir}")
    layout_df = parse_antsxbids_layout(bids_dir)
    
    # Filter for specific participant if requested
    if participant_label:
        # Handle cases where user might pass "sub-123" or just "123"
        target_id = participant_label.replace("sub-", "")
        layout_df = layout_df[layout_df['subjectID'] == target_id]
        print(f"Filtering for subject: {target_id}")

    # Filter for specific session if requested
    if session_label:
        # Handle cases where user might pass "ses-2023" or just "2023"
        target_ses = session_label.replace("ses-", "")
        layout_df = layout_df[layout_df['date'] == target_ses]
        print(f"Filtering for session: {target_ses}")

    if layout_df.empty:
        print("No valid subjects/sessions found.")
        return

    print(f"Found {len(layout_df)} unique sessions to process.")
    os.makedirs(output_dir, exist_ok=True)
    
    failures = []
    
    for idx, row in tqdm(layout_df.iterrows(), total=layout_df.shape[0]):
        success = process_session(
            row, 
            output_root=output_dir, 
            project_id=project,
            denoise_dti=denoise_dti,
            dti_moco='SyN'
        )
        if not success:
            failures.append(f"{row['subjectID']}_{row['date']}")

    if failures:
        print(f"Finished with {len(failures)} errors: {failures}")
    else:
        print("Processing complete successfully.")

@click.command()
@click.argument('bids_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--project', default='Project', help='Project ID string')
@click.option('--dl-weights', is_flag=True, help='Force download of ANTsPyMM/T1w templates and weights')
@click.option('--denoise/--no-denoise', default=True, help='Apply DTI denoising')
@click.option('--participant-label', help='Specific subject ID to process (e.g. 211239)')
@click.option('--session-label', help='Specific session ID to process (e.g. 20230405)')
@click.version_option(__version__)
def main(bids_dir, output_dir, project, dl_weights, denoise, participant_label, session_label):
    """
    ANTSXMM: Streamlined ANTsPyMM wrapper for ANTSXBIDS output.
    """
    
    # 1. Setup Data
    if dl_weights:
        print("Downloading templates and weights...")
        antspyt1w.get_data(force_download=True)
        antspymm.get_data(force_download=True)

    run_study(bids_dir, output_dir, project, denoise, participant_label, session_label)

if __name__ == '__main__': # pragma: no cover
    main()

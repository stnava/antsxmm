import click
import os
import sys
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

def run_study(bids_dir, output_dir, project, denoise_dti=True, 
       participant_label=None, session_label=None, separator='+', t1_run=None):
        
  print("Parsing BIDS layout from: {}".format(bids_dir))
  layout_df = parse_antsxbids_layout(bids_dir)
   
  # Filter for specific participant if requested
  if participant_label:
    layout_df = layout_df[layout_df['subjectID'] == participant_label]
    print("Filtering for subject: {}".format(participant_label))

  # Filter for specific session if requested
  if session_label:
    layout_df = layout_df[layout_df['date'] == session_label]
    print("Filtering for session: {}".format(session_label))

  if layout_df.empty:
    print("No valid subjects/sessions found.")
    return []

  print("Found {} unique sessions to process.".format(len(layout_df)))
  os.makedirs(output_dir, exist_ok=True)
   
  failures = []
   
  for idx, row in tqdm(layout_df.iterrows(), total=layout_df.shape[0]):
    result = process_session(
      row, 
      output_root=output_dir, 
      project_id=project,
      denoise_dti=denoise_dti,
      dti_moco='SyN',
      separator=separator,
      build_wide_table=True,
      t1_run_match=t1_run # Pass T1 filter
    )
     
    if not result['success']:
      failures.append("{}_{}".format(row['subjectID'], row['date']))

  if failures:
    print("Finished with {} errors: {}".format(len(failures), failures))
  else:
    print("Processing complete successfully.")
    
  return failures

@click.command()
@click.argument('bids_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--project', default='Project', help='Project ID string')
@click.option('--dl-weights', is_flag=True, help='Force download of ANTsPyMM/T1w templates and weights')
@click.option('--denoise/--no-denoise', default=True, help='Apply DTI denoising')
@click.option('--participant-label', help='Specific subject ID to process (e.g. sub-211239)')
@click.option('--session-label', help='Specific session ID to process (e.g. ses-20230405)')
@click.option('--t1-run', help='Specific T1 run string to match (e.g. r0002)')
@click.option('--separator', default='+', help='Character to separate filename components (default: +)')
@click.version_option(__version__)
def main(bids_dir, output_dir, project, dl_weights, denoise, participant_label, session_label, t1_run, separator):
  """
  ANTSXMM: Streamlined ANTsPyMM wrapper for ANTSXBIDS output.
  """
  print("antsxmm {}".format(__version__))
  # 1. Setup Data
  if dl_weights:
    print("Downloading templates and weights...")
    antspyt1w.get_data(force_download=True)
    antspymm.get_data(force_download=True)

  failures = run_study(bids_dir, output_dir, project, denoise, participant_label, session_label, separator, t1_run)

  if failures:
    sys.exit(1)

if __name__ == '__main__': # pragma: no cover
  main()
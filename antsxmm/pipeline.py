import os
import sys
import types
import logging
import warnings

import click
import pandas as pd
from tqdm import tqdm

# --- Logging Configuration ---
def setup_logging(verbose: bool):
    """Configures logging and silences known library noise."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=level,
        force=True
    )
    
    if not verbose:
        # Silence the specific scikit-learn random_state warnings from dependencies
        warnings.filterwarnings("ignore", category=UserWarning, module="antspyt1w")
        warnings.filterwarnings("ignore", category=UserWarning, module="antspymm")

# --- Optional Dependency Handling ---
try:
    import antspymm  # type: ignore
except ModuleNotFoundError:
    antspymm = types.SimpleNamespace()

try:
    import antspyt1w  # type: ignore
except ModuleNotFoundError:
    antspyt1w = types.SimpleNamespace()

try:
    from ._version import version as __version__
    from .bids import parse_antsxbids_layout
    from .core import process_session
except ImportError:
    # Fallback for local development/non-installed runs
    try:
        from antsxmm.bids import parse_antsxbids_layout
        from antsxmm.core import process_session
        from importlib.metadata import version
        __version__ = version("antsxmm")
    except Exception:
        __version__ = "0.0.0-dev"

# --- Shared Options Decorator ---
def pipeline_options(f):
    """Re-usable options for the pipeline to ensure consistent CLI behavior."""
    options = [
        click.option("--project", default="Project", help="Project ID string for file naming."),
        click.option("--dl-weights", is_flag=True, help="Force download of ANTsPyMM/T1w weights."),
        click.option("--denoise/--no-denoise", default=True, help="Apply DTI denoising."),
        click.option("--participant-label", help="Subject ID to process (e.g. sub-01)."),
        click.option("--session-label", help="Session ID to process (e.g. ses-01)."),
        click.option("--t1-run", help="Specific T1 run string to match (e.g. r01)."),
        click.option("--separator", default="+", help="Separator for filename components."),
        click.option("--input-manifest/--no-input-manifest", default=True, help="Write input JSON manifest."),
        click.option("--verbose/--no-verbose", default=False, help="Print detailed logs and show warnings.")
    ]
    for option in reversed(options):
        f = option(f)
    return f

# --- Core Logic ---
def run_study(
    bids_dir: str,
    output_dir: str,
    project: str,
    denoise_dti: bool = True,
    participant_label: str | None = None,
    session_label: str | None = None,
    separator: str = "+",
    t1_run: str | None = None,
    write_input_manifest: bool = True,
    verbose: bool = False,
) -> list[str]:
    setup_logging(verbose)
    logging.info(f"Parsing BIDS layout from: {bids_dir}")
    
    layout_df = parse_antsxbids_layout(bids_dir)

    if participant_label:
        layout_df = layout_df[layout_df["subjectID"].astype(str) == str(participant_label)]
        logging.info(f"Filtering for subject: {participant_label}")

    if session_label:
        # Cast to string to avoid pandas type mismatch with numeric session IDs
        layout_df = layout_df[layout_df["date"].astype(str) == str(session_label)]
        logging.info(f"Filtering for session: {session_label}")

    if layout_df.empty:
        logging.warning("No valid subjects/sessions found with provided filters.")
        return []

    logging.info(f"Found {len(layout_df)} unique sessions to process.")
    os.makedirs(output_dir, exist_ok=True)

    failures: list[str] = []

    for _, row in tqdm(layout_df.iterrows(), total=layout_df.shape[0], desc="Processing Sessions"):
        result = process_session(
            row,
            output_root=output_dir,
            project_id=project,
            denoise_dti=denoise_dti,
            dti_moco="SyN",
            separator=separator,
            build_wide_table=True,
            t1_run_match=t1_run,
            write_input_manifest=write_input_manifest,
            verbose=verbose,
        )

        if not result.get("success", False):
            failures.append(f"{row['subjectID']}_{row['date']}")

    if failures:
        logging.error(f"Finished with {len(failures)} errors: {failures}")
    else:
        logging.info("Processing completed successfully.")

    return failures

# --- CLI Implementation ---
@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(__version__)
def main():
    """
    🧠 ANTSXMM: Automated Multimodal Neuroimaging Pipeline.

    A streamlined wrapper for ANTsPyMM designed for ANTSXBIDS layouts.
    """
    pass

@main.command("run", short_help="Run the processing pipeline.")
@click.argument("bids_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@pipeline_options
def run_cmd(bids_dir, output_dir, **kwargs):
    """
    🚀 Run the full processing pipeline on a BIDS directory.
    
    BIDS_DIR: Path to the root of your BIDS dataset.\n
    OUTPUT_DIR: Path to save processed 'pymm' results.
    """
    _run_pipeline_logic(bids_dir, output_dir, **kwargs)

@main.command("tree", short_help="Visualize output directory structure.")
@click.argument("path", type=click.Path(exists=True))
@click.option("--create", is_flag=True, help="Actually create the predicted directory structure.")
def tree_cmd(path: str, create: bool) -> None:
    """🌲 Predict and visualize the output directory tree for a subject."""
    from pathlib import Path as _Path
    try:
        from .tree import predict_tree
    except ImportError:
        from antsxmm.tree import predict_tree

    project, subject, tree = predict_tree(path)

    click.secho(f"Project: {project} | Subject: {subject}", fg="cyan", bold=True)
    for ses, runs in tree.items():
        print(f"  {ses}/")
        seen: set[tuple[str, str]] = set()
        for modality, run in runs:
            if (modality, run) in seen: continue
            seen.add((modality, run))
            print(f"    {modality}/")
            print(f"      {run}/")
            if create:
                d = _Path("pymm") / project / subject / ses / modality / run
                d.mkdir(parents=True, exist_ok=True)

@main.command("validate", short_help="Check processed outputs.")
@click.argument("path", type=click.Path(exists=True))
@click.option("--pymm-dir", default="pymm", help="Path to pymm output root.")
def validate_cmd(path: str, pymm_dir: str) -> None:
    """✅ Validate processed outputs against expected structure and files."""
    try:
        from .validate import validate_project
    except ImportError:
        from antsxmm.validate import validate_project

    results = validate_project(path, pymm_dir=pymm_dir)
    for session_key, res in results.items():
        click.secho(f"Session: {session_key}", bold=True)
        if res.missing:
            click.secho(f"  Missing: {len(res.missing)} files", fg="red")
        if res.ok:
            click.secho(f"  OK: {len(res.ok)} files", fg="green")
        print("")

def _run_pipeline_logic(
    bids_dir: str,
    output_dir: str,
    project: str,
    dl_weights: bool,
    denoise: bool,
    participant_label: str | None,
    session_label: str | None,
    t1_run: str | None,
    separator: str,
    input_manifest: bool,
    verbose: bool,
) -> None:
    """Internal logic to bridge CLI and execution."""
    setup_logging(verbose)
    logging.info(f"antsxmm version {__version__}")

    if dl_weights:
        if not hasattr(antspyt1w, "get_data") or not hasattr(antspymm, "get_data"):
            logging.error("--dl-weights requires antspyt1w and antspymm to be installed.")
            sys.exit(1)
        logging.info("Downloading templates and weights (this may take a while)...")
        antspyt1w.get_data(force_download=True)
        antspymm.get_data(force_download=True)

    failures = run_study(
        bids_dir=bids_dir,
        output_dir=output_dir,
        project=project,
        denoise_dti=denoise,
        participant_label=participant_label,
        session_label=session_label,
        separator=separator,
        t1_run=t1_run,
        write_input_manifest=input_manifest,
        verbose=verbose,
    )

    if failures:
        sys.exit(1)

def entry_point():
    """Allows calling 'antsxmm BIDS OUT' without the 'run' keyword."""
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        # If first arg is not a command, not a help flag, and looks like a path/string
        if arg1 not in main.commands and arg1 not in ['-h', '--help', '--version']:
            sys.argv.insert(1, 'run')
    main()

if __name__ == "__main__":
    entry_point()
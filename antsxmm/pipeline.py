import os
import sys
import types
import logging
import warnings
import json
from pathlib import Path

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
    from .core import process_session, compute_input_fingerprint
except ImportError:
    # Fallback for local development/non-installed runs
    try:
        from antsxmm.bids import parse_antsxbids_layout
        from antsxmm.core import process_session, compute_input_fingerprint
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
        click.option("--resume/--no-resume", default=True, help="Skip sessions already complete and unchanged."),
        click.option("--force", is_flag=True, help="Re-run sessions even if already complete."),
        click.option("--rerun-failed", is_flag=True, help="Only run sessions that previously failed (or have no status)."),
        click.option("--dry-run", is_flag=True, help="Print the execution plan without running processing."),
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
    resume: bool = True,
    force: bool = False,
    rerun_failed: bool = False,
    dry_run: bool = False,
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
    logging.info(f"Found {len(layout_df)} unique sessions to process.")
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    failures: list[str] = []

    def _status_path_for(sub: str, ses: str) -> Path:
        return Path(output_dir) / project / sub / ses / '.antsxmm_status.json'

    def _load_status(p: Path) -> dict | None:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return None
        return None

    planned = 0
    skipped = 0
    to_run = 0

    for _, row in tqdm(layout_df.iterrows(), total=layout_df.shape[0], desc="Processing Sessions"):
        row_dict = row.to_dict()
        sub = str(row_dict.get('subjectID'))
        ses = str(row_dict.get('date'))

        # Decide if we should run this session
        status_path = _status_path_for(sub, ses)
        status = _load_status(status_path)

        fingerprint = compute_input_fingerprint(row_dict, t1_run_match=t1_run)

        should_run = True
        reason = 'run'

        if force:
            should_run = True
            reason = 'force'
        elif rerun_failed:
            if status is None:
                should_run = True
                reason = 'no_status'
            else:
                should_run = not bool(status.get('success', False))
                reason = 'rerun_failed' if should_run else 'already_success'
        elif resume and status is not None and bool(status.get('success', False)):
            prev = status.get('input_fingerprint', {}) or {}
            if prev.get('hash') and fingerprint.get('hash') and prev.get('hash') == fingerprint.get('hash'):
                should_run = False
                reason = 'resume_skip'
            else:
                should_run = True
                reason = 'inputs_changed'

        planned += 1
        if should_run:
            to_run += 1
        else:
            skipped += 1

        if dry_run:
            click.echo(f"PLAN {sub}_{ses}: {'RUN' if should_run else 'SKIP'} ({reason})")
            continue

        if not should_run:
            continue

        result = process_session(
            row_dict,
            output_root=output_dir,
            project_id=project,
            denoise_dti=denoise_dti,
            dti_moco="SyN",
            separator=separator,
            build_wide_table=True,
            t1_run_match=t1_run,
            write_input_manifest=write_input_manifest,
            verbose=verbose,
            tool_version=__version__,
            resume_mode=(
                'force' if force else
                'rerun_failed' if rerun_failed else
                'resume' if resume else
                'no_resume'
            ),
        )

        if not result.get("success", False):
            failures.append(f"{sub}_{ses}")

    if dry_run:
        click.echo(f"Plan summary: sessions={planned} run={to_run} skip={skipped}")
        return []

    if failures:
        # Keep a human-readable summary on stdout for library use and for tests.
        # (Logging may be redirected or suppressed by callers.)
        print(f"Finished with {len(failures)} errors")
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
    # Print a concrete, copy-pastable path layout rooted at the default output dir.
    print("pymm/")
    print(f"  {project}/")
    print(f"    {subject}/")
    for ses, runs in tree.items():
        print(f"      {ses}/")
        seen: set[tuple[str, str]] = set()
        for modality, run in runs:
            if (modality, run) in seen: continue
            seen.add((modality, run))
            print(f"        {modality}/")
            print(f"          {run}/")
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
    resume: bool,
    force: bool,
    rerun_failed: bool,
    dry_run: bool,
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
        resume=resume,
        force=force,
        rerun_failed=rerun_failed,
        dry_run=dry_run,
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

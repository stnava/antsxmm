import os
import sys
import logging
import warnings
import json
from pathlib import Path

try:
    from .environment import apply_default_environment, format_environment_policy_for_log, get_effective_environment_policy
    from .runtime_imports import import_optional_module
except ImportError:
    from antsxmm.environment import apply_default_environment, format_environment_policy_for_log, get_effective_environment_policy
    from antsxmm.runtime_imports import import_optional_module

import click
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

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    
    if not verbose:
        # Silence the specific scikit-learn random_state warnings from dependencies
        warnings.filterwarnings("ignore", category=UserWarning, module="antspyt1w")
        warnings.filterwarnings("ignore", category=UserWarning, module="antspymm")

# --- Optional Dependency Handling ---
antspymm = import_optional_module("antspymm")
antspyt1w = import_optional_module("antspyt1w")

try:
    from ._version import version as __version__
    from .bids import parse_antsxbids_layout
    from .core import process_session, compute_input_fingerprint
    from .execution_plan import build_execution_plan
    from .diagnostics import diagnose_bids_tree, write_study_diagnostics_json, format_study_diagnostics_summary
except ImportError:
    # Fallback for local development/non-installed runs
    try:
        from antsxmm.bids import parse_antsxbids_layout
        from antsxmm.core import process_session, compute_input_fingerprint
        from antsxmm.execution_plan import build_execution_plan
        from antsxmm.diagnostics import diagnose_bids_tree, write_study_diagnostics_json, format_study_diagnostics_summary
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



def _planned_output_markers(output_dir: str, project: str, row_dict: dict) -> list[Path]:
    """Return deterministic output markers expected for a successful session."""
    try:
        plan = build_execution_plan(row_dict, output_root=output_dir, project_id=project)
    except Exception:
        return []

    markers: list[Path] = []
    for unit in plan:
        prefix = Path(unit.output_prefix)
        marker = prefix.parent / f"{prefix.name}+mmwide.csv"
        markers.append(marker)
    return markers


def _session_outputs_complete(output_dir: str, project: str, row_dict: dict) -> bool:
    """A prior success status is only reusable if planned outputs still exist on disk."""
    markers = _planned_output_markers(output_dir, project, row_dict)
    if not markers:
        return False
    return all(marker.exists() for marker in markers)

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
    env_payload = apply_default_environment()
    for line in format_environment_policy_for_log({
        "thread_count": env_payload["thread_count"],
        "thread_source": env_payload["thread_source"],
        "effective": env_payload["effective"],
    }):
        logging.info(line)
    logging.info(f"Parsing BIDS layout from: {bids_dir}")
    
    layout_df = parse_antsxbids_layout(bids_dir)

    normalized_participant_label = str(participant_label).rstrip('/\\') if participant_label else None
    if normalized_participant_label:
        logging.info(f"Filtering for subject: {normalized_participant_label}")
    if session_label:
        logging.info(f"Filtering for session: {session_label}")

    required_columns = {"subjectID", "date"}
    has_required_columns = hasattr(layout_df, "columns") and required_columns.issubset(set(layout_df.columns))

    if has_required_columns and normalized_participant_label:
        layout_df = layout_df[layout_df["subjectID"].astype(str) == normalized_participant_label]

    if has_required_columns and session_label:
        # Cast to string to avoid pandas type mismatch with numeric session IDs
        layout_df = layout_df[layout_df["date"].astype(str) == str(session_label)]

    if (not has_required_columns) or layout_df.empty:
        diagnostics = diagnose_bids_tree(bids_dir)
        diagnostics["requested_filters"] = {
            "participant_label": normalized_participant_label,
            "session_label": str(session_label) if session_label else None,
        }
        if hasattr(layout_df, "columns"):
            diagnostics["parsed_layout_columns"] = list(layout_df.columns)
        diagnostics["parsed_layout_row_count"] = int(layout_df.shape[0]) if hasattr(layout_df, "shape") else 0
        diagnostics_path = None
        if not dry_run:
            diagnostics_path = write_study_diagnostics_json(output_dir, project, diagnostics)
            diagnostics["diagnostics_path"] = diagnostics_path
        for line in format_study_diagnostics_summary(
            diagnostics,
            participant_label=normalized_participant_label,
            session_label=str(session_label) if session_label else None,
        ):
            logging.warning(line)
            print(f"[DIAGNOSTIC] {line}")
        if diagnostics_path:
            logging.warning(f"Study input diagnostics written to: {diagnostics_path}")
            print(f"[DIAGNOSTIC] Study input diagnostics written to: {diagnostics_path}")
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
            hashes_match = bool(prev.get('hash') and fingerprint.get('hash') and prev.get('hash') == fingerprint.get('hash'))
            if hashes_match and _session_outputs_complete(output_dir, project, row_dict):
                should_run = False
                reason = 'resume_skip'
            elif hashes_match:
                should_run = True
                reason = 'outputs_missing'
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



@main.command("aggregate", short_help="Aggregate mmwidemerged tables into one study table.")
@click.argument("root", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output table path (.csv or .parquet).",
)
@click.option(
    "--pattern",
    default="*mmwidemerged*.csv",
    show_default=True,
    help="Recursive glob pattern used to discover per-run merged tables under ROOT. The default also matches duplicate copies like *mmwidemerged 2.csv.",
)
@click.option(
    "--state",
    "state_path",
    type=click.Path(path_type=Path),
    help="Incremental state file. Defaults to <output>.state.json.",
)
@click.option(
    "--rejects",
    "rejects_path",
    type=click.Path(path_type=Path),
    help="CSV file listing rejected inputs. Defaults to <output>.rejects.csv.",
)
@click.option(
    "--prefer",
    type=click.Choice(["processed-first", "pymm-first", "newest", "largest", "error"], case_sensitive=False),
    default="processed-first",
    show_default=True,
    help="Duplicate-resolution policy when multiple files map to the same project/subject/session/modality/run entity.",
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    show_default=True,
    help="Reuse the existing aggregate and only re-read changed entities when state is available.",
)
def aggregate_cmd(
    root: Path,
    output: Path,
    pattern: str,
    state_path: Path | None,
    rejects_path: Path | None,
    prefer: str,
    incremental: bool,
) -> None:
    """Aggregate discovered *mmwidemerged*.csv files into one study-level table.

    This supports both nested layouts like ROOT/pymm/<project>/sub-.../ses-.../
    <modality>/run-.../*.csv and flat drop-zones that only contain the merged
    CSV filenames themselves. Duplicate Finder-style copies such as
    `...+mmwidemerged 2.csv` are normalized back to the same entity.

    ROOT is a directory that contains one or more source roots such as pymm/
    and Processed/. Each discovered file must resolve to a unique entity using
    project_id, subject_id, session_id, modality, and run_id from the path or
    the file name.

    The output table keeps one row per entity after duplicate resolution. When
    incremental mode is enabled, antsxmm stores a lightweight state file so
    unchanged entities can be reused on later runs.
    """
    try:
        from .aggregate import aggregate_merged_tables
    except ImportError:
        from antsxmm.aggregate import aggregate_merged_tables

    result = aggregate_merged_tables(
        root=root,
        output=output,
        pattern=pattern,
        state_path=state_path,
        rejects_path=rejects_path,
        incremental=incremental,
        prefer=prefer.lower(),
    )

    click.echo(
        "aggregate "
        f"scanned={result.scanned} "
        f"read={result.read} "
        f"rows_written={result.rows_written} "
        f"rejected={result.rejected} "
        f"incremental={'yes' if result.incremental else 'no'} "
        f"reused_existing={'yes' if result.reused_existing else 'no'}"
    )
    click.echo(f"output={result.output_path}")
    click.echo(f"state={result.state_path}")
    click.echo(f"rejects={result.rejects_path}")

@main.command("validate", short_help="Validate processed outputs against a BIDS project.")
@click.argument("input_bids_project", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--participant-label",
    multiple=True,
    help="Restrict validation to one or more subject IDs, e.g. --participant-label sub-0102.",
)
@click.option("--summary-only", is_flag=True, help="Print only study-level summary tables.")
@click.option("--issues-only", is_flag=True, help="Only print per-run rows with non-OK status.")
@click.option("--all-rows", is_flag=True, help="Print all per-run rows, including OK rows.")
@click.option("--report-json", type=click.Path(path_type=Path), help="Write a machine-readable JSON validation report.")
def validate_cmd(
    input_bids_project: Path,
    output_dir: Path,
    participant_label: tuple[str, ...],
    summary_only: bool,
    issues_only: bool,
    all_rows: bool,
    report_json: Path | None,
) -> None:
    """Validate expected antsxmm outputs and report study-level and per-run results.

    INPUT_BIDS_PROJECT is one dataset directory such as BIDS/breacher.
    OUTPUT_DIR is the processed antsxmm root such as pymm or Processed.
    """
    try:
        from .validate import (
            validate_project,
            summarize_results,
            build_session_modality_table,
            build_missing_percentage_table,
            build_issue_code_summary,
            build_validation_report,
            write_report_json,
        )
    except ImportError:
        from antsxmm.validate import (
            validate_project,
            summarize_results,
            build_session_modality_table,
            build_missing_percentage_table,
            build_issue_code_summary,
            build_validation_report,
            write_report_json,
        )

    validation_report = build_validation_report(
        input_bids_project,
        output_dir,
        participant_labels=participant_label or None,
    )
    results = validation_report.legacy_results
    if report_json is not None:
        report_path = write_report_json(validation_report, report_json)
    summary = summarize_results(results)
    rows = build_session_modality_table(results)
    missing_pct_rows = build_missing_percentage_table(results)
    issue_rows = build_issue_code_summary(results)

    show_all_rows = bool(all_rows or (participant_label and not issues_only and not summary_only))
    show_only_issues = bool(issues_only or (not show_all_rows and not summary_only))

    click.secho("Validation summary", fg="cyan", bold=True)
    click.echo(f"  Sessions checked: {summary.session_count}")
    click.echo(f"  Clean sessions: {summary.clean_session_count}")
    click.echo(f"  Sessions with issues: {summary.affected_session_count}")
    click.echo(f"  OK directories: {summary.ok_count}")
    click.echo(f"  Missing directories: {summary.missing_count}")
    click.echo(f"  Unexpected directories: {summary.unexpected_count}")
    click.echo(f"  Missing mmwide.csv files: {summary.missing_mmwide_count}")
    click.echo(f"  Invalid mmwide.csv files: {summary.invalid_mmwide_count}")
    click.echo(f"  Missing status files: {summary.missing_status_count}")
    click.echo("")

    click.secho("Missing percentage table", fg="cyan", bold=True)
    click.echo(f"{'Modality':<20} {'Expected':>8} {'DirOK':>8} {'DirMiss':>8} {'DirMiss%':>9} {'CSVOK':>8} {'CSVMiss':>8} {'CSVMiss%':>9} {'CSVBad':>8} {'CSVBad%':>8}")
    click.echo(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8} {'-' * 8}")
    for row in missing_pct_rows:
        click.echo(
            f"{row.modality:<20} {row.expected_count:>8} {row.present_dir_count:>8} {row.missing_dir_count:>8} {row.missing_dir_pct:>8.1f}% {row.present_mmwide_count:>8} {row.missing_mmwide_count:>8} {row.missing_mmwide_pct:>8.1f}% {row.invalid_mmwide_count:>8} {row.invalid_mmwide_pct:>7.1f}%"
        )

    click.echo("")
    click.secho("Issue code summary", fg="cyan", bold=True)
    if issue_rows:
        click.echo(f"{'code':<28} {'count':>8}")
        click.echo(f"{'-' * 28} {'-' * 8}")
        for row in issue_rows:
            click.echo(f"{row.code:<28} {row.count:>8}")
    else:
        click.echo("No issues detected.")

    if report_json is not None:
        click.echo("")
        click.echo(f"JSON report: {report_path}")

    if summary_only:
        return

    filtered_rows = [row for row in rows if row.status != "OK"] if show_only_issues else rows
    click.echo("")
    click.secho("Per-run validation table", fg="cyan", bold=True)
    click.echo(
        f"{'subject_id':<14} {'session_id':<20} {'modality':<18} {'run_id':<10} {'status':<12} expected_mmwide_csv"
    )
    click.echo(
        f"{'-' * 14} {'-' * 20} {'-' * 18} {'-' * 10} {'-' * 12} {'-' * 20}"
    )
    for row in filtered_rows:
        status_color = "green" if row.status == "OK" else ("yellow" if row.status == "MISSING_CSV" else "red")
        click.secho(
            f"{row.subject_id:<14} {row.session_id:<20} {row.modality:<18} {row.run_id:<10} {row.status:<12} {row.expected_mmwide_csv or '-'}",
            fg=status_color,
        )

    for session_key, res in results.items():
        has_issues = bool(res.missing or res.unexpected or res.missing_mmwide_files or res.invalid_mmwide_files or res.missing_status_files)
        if not has_issues:
            continue
        click.echo("")
        click.secho(f"Session: {session_key}", bold=True)
        if res.missing_modalities:
            click.secho(f"  Missing modalities: {', '.join(res.missing_modalities)}", fg="red")
        if res.missing:
            click.secho(f"  Missing directories: {len(res.missing)}", fg="red")
            for item in res.missing:
                click.echo(f"    - {item}")
        if res.unexpected:
            click.secho(f"  Unexpected directories: {len(res.unexpected)}", fg="yellow")
            for item in res.unexpected:
                click.echo(f"    - {item}")
        if res.missing_mmwide_files:
            click.secho(f"  Missing mmwide.csv files: {len(res.missing_mmwide_files)}", fg="red")
            for item in res.missing_mmwide_files:
                click.echo(f"    - {item}")
        if res.invalid_mmwide_files:
            click.secho(f"  Invalid mmwide.csv files: {len(res.invalid_mmwide_files)}", fg="red")
            for item in res.invalid_mmwide_files:
                click.echo(f"    - {item}")
        if res.missing_status_files:
            click.secho(f"  Missing status files: {len(res.missing_status_files)}", fg="yellow")
            for item in res.missing_status_files:
                click.echo(f"    - {item}")
        if res.ok:
            click.secho(f"  OK directories: {len(res.ok)}", fg="green")
        click.echo("")

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
    for line in format_environment_policy_for_log(get_effective_environment_policy()):
        logging.info(line)

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
    return main()

if __name__ == "__main__":
    entry_point()

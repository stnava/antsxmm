import os
import sys
import types

import click
import pandas as pd
from tqdm import tqdm


try:  # optional dependency for lightweight installs / unit tests
    import antspymm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    antspymm = types.SimpleNamespace()

try:  # optional dependency for lightweight installs / unit tests
    import antspyt1w  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    antspyt1w = types.SimpleNamespace()


try:
    from ._version import version as __version__
    from .bids import parse_antsxbids_layout
    from .core import process_session
except ImportError:  # pragma: no cover
    from antsxmm.bids import parse_antsxbids_layout
    from antsxmm.core import process_session

    try:
        from importlib.metadata import version

        __version__ = version("antsxmm")
    except Exception:
        __version__ = "0.0.0-dev"


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
    print(f"Parsing BIDS layout from: {bids_dir}")
    layout_df = parse_antsxbids_layout(bids_dir)

    if participant_label:
        layout_df = layout_df[layout_df["subjectID"] == participant_label]
        print(f"Filtering for subject: {participant_label}")

    if session_label:
        layout_df = layout_df[layout_df["date"] == session_label]
        print(f"Filtering for session: {session_label}")

    if layout_df.empty:
        print("No valid subjects/sessions found.")
        return []

    print(f"Found {len(layout_df)} unique sessions to process.")
    os.makedirs(output_dir, exist_ok=True)

    failures: list[str] = []

    for _, row in tqdm(layout_df.iterrows(), total=layout_df.shape[0]):
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
        print(f"Finished with {len(failures)} errors: {failures}")
    else:
        print("Processing complete successfully.")

    return failures


@click.group(
    invoke_without_command=True,
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option("--project", default="Project", help="Project ID string")
@click.option(
    "--dl-weights",
    is_flag=True,
    help="Force download of ANTsPyMM/T1w templates and weights",
)
@click.option("--denoise/--no-denoise", default=True, help="Apply DTI denoising")
@click.option(
    "--participant-label", help="Specific subject ID to process (e.g. sub-211239)"
)
@click.option("--session-label", help="Specific session ID to process (e.g. ses-20230405)")
@click.option("--t1-run", help="Specific T1 run string to match (e.g. r0002)")
@click.option(
    "--separator",
    default="+",
    help="Character to separate filename components (default: +)",
)
@click.option(
    "--input-manifest/--no-input-manifest",
    default=True,
    help="Write a per-session JSON listing exactly which NIfTI inputs will be processed",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Print discovered files and selected inputs per session",
)
@click.version_option(__version__)
@click.pass_context
def main(
    ctx: click.Context,
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
    """antsxmm CLI.

    Commands:
      - (default) run: antsxmm <BIDS_DIR> <OUTPUT_DIR> [options]
      - tree: antsxmm tree <SUBJECT_DIR> [--create]
      - validate: antsxmm validate <PROJECT_DIR> [--pymm-dir pymm/]
    """
    if ctx.invoked_subcommand is None:
        if len(ctx.args) != 2:
            raise click.UsageError("Expected: antsxmm <BIDS_DIR> <OUTPUT_DIR> [options]")
        bids_dir, output_dir = ctx.args
        _run_pipeline(
            bids_dir=bids_dir,
            output_dir=output_dir,
            project=project,
            dl_weights=dl_weights,
            denoise=denoise,
            participant_label=participant_label,
            session_label=session_label,
            t1_run=t1_run,
            separator=separator,
            input_manifest=input_manifest,
            verbose=verbose,
        )


@main.command("run")
@click.argument("bids_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.pass_context
def run_cmd(ctx: click.Context, bids_dir: str, output_dir: str) -> None:
    """Run the antsxmm pipeline (explicit command form)."""
    params = ctx.parent.params if ctx.parent else {}
    _run_pipeline(bids_dir=bids_dir, output_dir=output_dir, **params)


@main.command("tree")
@click.argument("path", type=click.Path(exists=True))
@click.option("--create", is_flag=True, help="Create predicted directory structure")
def tree_cmd(path: str, create: bool) -> None:
    from pathlib import Path as _Path
    from .tree import predict_tree

    project, subject, tree = predict_tree(path)

    print("pymm/")
    print(f"  {project}/")
    print(f"    {subject}/")

    for ses, runs in tree.items():
        print(f"      {ses}/")

        seen: set[tuple[str, str]] = set()

        for modality, run in runs:
            if (modality, run) in seen:
                continue
            seen.add((modality, run))

            print(f"        {modality}/")
            print(f"          {run}/")

            if create:
                d = _Path("pymm") / project / subject / ses / modality / run
                d.mkdir(parents=True, exist_ok=True)


@main.command("validate")
@click.argument("path", type=click.Path(exists=True))
@click.option("--pymm-dir", default="pymm", help="Path to pymm output root")
def validate_cmd(path: str, pymm_dir: str) -> None:
    from .validate import validate_project

    results = validate_project(path, pymm_dir=pymm_dir)
    for session_key, res in results.items():
        print(f"Session: {session_key}")

        if res.missing:
            print("Missing:")
            for m in res.missing:
                print(f"  {m}")
        if res.unexpected:
            print("Unexpected:")
            for u in res.unexpected:
                print(f"  {u}")
        if res.ok:
            print("OK:")
            for o in res.ok:
                print(f"  {o}")
        print("")


def _run_pipeline(
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
    """ANTSXMM: Streamlined ANTsPyMM wrapper for ANTSXBIDS output."""

    print(f"antsxmm {__version__}")

    if dl_weights:
        if not hasattr(antspyt1w, "get_data") or not hasattr(antspymm, "get_data"):
            raise ModuleNotFoundError(
                "--dl-weights requires antspyt1w and antspymm to be installed."
            )
        print("Downloading templates and weights...")
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




if __name__ == "__main__":  # pragma: no cover
    main()

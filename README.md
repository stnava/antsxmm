# ANTsXMM

ANTsXMM is a BIDS-oriented orchestration layer for [ANTsPyMM](https://github.com/ANTsX/ANTsPyMM). It discovers multimodal inputs from study / subject / session trees, builds a deterministic execution plan, stages processing safely, and writes reproducible per-session outputs and wide-table artifacts.

![The ANTsXMM framework](docs/antsxmm_infographic.png)

Documentation of functions is available [here](https://htmlpreview.github.io/?https://raw.githubusercontent.com/stnava/antsxmm/main/docs/antsxmm.html).

## Recent progress

Recent work in this repository focused on making ingestion and execution more robust for real-world BIDS-like datasets:

- added first-class `NM2DMT` modality support from `*_NM.nii.gz` inputs
- hardened deterministic prefix generation so `mm_csv` no longer fails when an expected modality prefix column is missing
- added acceptance of legacy run tokens such as `r0001`, with normalization to canonical internal run ids such as `run-01`
- expanded discovery and tree/planning support for `NM2DMT`, `DTI`, `rsfMRI`, and `pet3d`
- improved subject filtering so common CLI mistakes such as a trailing slash on `--participant-label` do not silently exclude valid sessions
- added process-wide environment default management for thread-sensitive libraries while preserving user, scheduler, and container overrides
- extended regression coverage around modality mapping, deterministic dataframe generation, tree prediction, participant/session filtering, and execution seams

## Installation

```bash
pip install .
pip install ".[test]"
```

## Current CLI shape

```bash
antsxmm run <BIDS_DIR> <OUTPUT_DIR> --project <PROJECT>
antsxmm tree <PATH>
antsxmm validate <INPUT_BIDS_PROJECT> <OUTPUT_DIR>
antsxmm aggregate <ROOT> --output <AGGREGATE.csv>
```

A compatibility entry path is also supported, so this still works:

```bash
antsxmm <BIDS_DIR> <OUTPUT_DIR> --project <PROJECT>
```

## Usage

### Inspect a dataset or subject

```bash
antsxmm tree BIDS/PPMI
antsxmm tree BIDS/PPMI/sub-182341
```

### Run a single subject/session

```bash
antsxmm run BIDS/PPMI pymm --project PPMI \
  --participant-label sub-182341 \
  --session-label ses-20230111
```

### Run a study

```bash
antsxmm run BIDS/PPMI pymm --project PPMI
```

### Dry-run the execution plan

```bash
antsxmm run BIDS/PPMI pymm --project PPMI --dry-run --verbose
```

### Aggregate per-run merged tables

```bash
antsxmm aggregate study_root --output study_root/aggregate.csv
```

`aggregate` recursively discovers `*mmwidemerged.csv` files below `ROOT`, resolves
study entities from the directory layout and filename, and writes one study-level
row per `(project_id, subject_id, session_id, modality, run_id)` entity. By
default it prefers `Processed/` over `pymm/` when duplicates exist and keeps an
incremental state file next to the output so later runs only re-read changed
entities.

## Environment defaults and cluster usage

ANTsXMM now applies runtime defaults in a dedicated CLI bootstrap module before the heavier pipeline module and optional imaging dependencies are imported. This lets all child processes inherit the same environment while still respecting explicit user or scheduler settings. Programmatic callers that invoke pipeline functions directly remain supported; in that case antsxmm applies the same defaults at runtime before processing begins.

The following variables are managed when they are not already set:

- `TF_NUM_INTEROP_THREADS`
- `TF_NUM_INTRAOP_THREADS`
- `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `MPLBACKEND=Agg`

Thread count selection priority is:

1. `ANTSXMM_THREADS`
2. `SLURM_CPUS_PER_TASK`
3. fallback default `8`

Examples:

```bash
export ANTSXMM_THREADS=8
antsxmm run BIDS/PPMI pymm --project PPMI
```

```bash
sbatch --cpus-per-task=8 ...
```

When `--verbose` is enabled, antsxmm logs the effective environment policy and whether each variable was preserved or defaulted.

## SLURM helper scripts

The repository includes starter scripts in `scripts/`:

- `submit_antsxmm_bids.sh`
- `run_antsxmm_subject.slurm`
- `zipit.sh`

These are intended as cluster starting points and may require site-specific adjustment for account, partition, module loading, conda activation, memory, and CPU policy.

## Testing

```bash
pytest --cov=antsxmm --cov-report=term-missing tests/
```

## Versioning

Create a new semantic version tag, for example:

```bash
git tag v0.2.0
```

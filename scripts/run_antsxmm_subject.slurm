#!/usr/bin/env bash
#SBATCH --job-name=antsxmm-subject
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  run_antsxmm_subject.slurm <BIDS_SUBJECT_DIR|BIDS_PROJECT_DIR> <OUTPUT_ROOT> [SUBJECT_LABEL] [SESSION_LABEL] [PROJECT_ID]

Examples:
  run_antsxmm_subject.slurm /data/BIDS/breacher/sub-001 /data/pymm
  run_antsxmm_subject.slurm /data/BIDS/FPA sub-001 /data/pymm
  run_antsxmm_subject.slurm /data/BIDS/FPA /data/pymm sub-001 ses-20230111 FPA

Behavior:
  - If the first argument is a subject directory (e.g. .../breacher/sub-001), the script derives:
      BIDS_PROJECT_DIR=.../breacher
      SUBJECT_LABEL=sub-001
      PROJECT_ID=breacher
  - If the first argument is a project directory (e.g. .../FPA), then SUBJECT_LABEL must be provided
    as the third argument and PROJECT_ID defaults to the basename of the project directory unless
    the optional PROJECT_ID argument is given explicitly.
USAGE
  exit 2
}

[[ $# -lt 2 ]] && usage

INPUT_PATH="$1"
OUTPUT_ROOT="$2"
SUBJECT_LABEL="${3:-}"
SESSION_LABEL="${4:-}"
PROJECT_ID="${5:-}"

if [[ ! -d "$INPUT_PATH" ]]; then
  echo "ERROR: input path does not exist or is not a directory: $INPUT_PATH" >&2
  exit 2
fi

basename_no_trailing() {
  local p="$1"
  p="${p%/}"
  basename "$p"
}

leaf="$(basename_no_trailing "$INPUT_PATH")"

if [[ "$leaf" == sub-* ]]; then
  BIDS_PROJECT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
  : "${SUBJECT_LABEL:=$leaf}"
  if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="$(basename_no_trailing "$BIDS_PROJECT_DIR")"
  fi
else
  BIDS_PROJECT_DIR="$(cd "$INPUT_PATH" && pwd)"
  if [[ -z "$SUBJECT_LABEL" ]]; then
    echo "ERROR: SUBJECT_LABEL is required when INPUT_PATH is a project directory" >&2
    usage
  fi
  if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="$(basename_no_trailing "$BIDS_PROJECT_DIR")"
  fi
fi

cmd=(
  antsxmm run
  "$BIDS_PROJECT_DIR"
  "$OUTPUT_ROOT"
  --project "$PROJECT_ID"
  --participant-label "$SUBJECT_LABEL"
)

if [[ -n "$SESSION_LABEL" ]]; then
  cmd+=(--session-label "$SESSION_LABEL")
fi

printf 'Running:' >&2
printf ' %q' "${cmd[@]}" >&2
printf '\n' >&2
exec "${cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash submit_antsxmm_bids.sh /path/to/BIDS /path/to/output
#
# Example:
#   bash submit_antsxmm_bids.sh /data/BIDS /data/antsxmm_out

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <BIDS_ROOT> <OUT_ROOT>" >&2
  exit 2
fi

BIDS_ROOT="$(cd "$1" && pwd)"
OUT_ROOT="$2"

# ---- site-specific knobs ----
PARTITION="${PARTITION:-compute}"
ACCOUNT="${ACCOUNT:-}"
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-FAIL}"
CONDA_ENV="${CONDA_ENV:-ants}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/slurm_logs}"
JOB_NAME="${JOB_NAME:-antsxmm}"
MAX_PARALLEL="${MAX_PARALLEL:-12}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

SUBJECT_LIST="${OUT_ROOT}/subjects_all.txt"

# Build study/subject manifest:
# Each line is:
#   STUDY<TAB>SUBJECT_DIR<TAB>SUBJECT_ID
find "${BIDS_ROOT}" -mindepth 2 -maxdepth 2 -type d -name 'sub-*' | sort | \
awk -F/ '
{
  study=$(NF-1);
  subj=$NF;
  print study "\t" $0 "\t" subj
}' > "${SUBJECT_LIST}"

N=$(wc -l < "${SUBJECT_LIST}" | tr -d ' ')
if [[ "${N}" -eq 0 ]]; then
  echo "No subject directories found under ${BIDS_ROOT}" >&2
  exit 1
fi

ARRAY_SPEC="0-$((N-1))%${MAX_PARALLEL}"

SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --time="${TIME}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --array="${ARRAY_SPEC}"
  --output="${LOG_ROOT}/%x_%A_%a.out"
  --error="${LOG_ROOT}/%x_%A_%a.err"
  --export=ALL,BIDS_ROOT="${BIDS_ROOT}",OUT_ROOT="${OUT_ROOT}",SUBJECT_LIST="${SUBJECT_LIST}",CONDA_ENV="${CONDA_ENV}"
)

if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi

if [[ -n "${MAIL_USER}" ]]; then
  SBATCH_ARGS+=(--mail-user="${MAIL_USER}" --mail-type="${MAIL_TYPE}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_antsxmm_subject.slurm"

echo "Submitting ${N} subjects from ${BIDS_ROOT}"
sbatch "${SBATCH_ARGS[@]}" "${WORKER}"

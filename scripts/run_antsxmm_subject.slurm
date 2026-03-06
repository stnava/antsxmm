#!/usr/bin/env bash
#SBATCH --nodes=1

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
: "${SUBJECT_LIST:?missing SUBJECT_LIST}"
: "${BIDS_ROOT:?missing BIDS_ROOT}"
: "${OUT_ROOT:?missing OUT_ROOT}"
: "${CONDA_ENV:?missing CONDA_ENV}"

line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SUBJECT_LIST}")"
if [[ -z "${line}" ]]; then
  echo "No manifest entry for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r STUDY SUBJECT_DIR SUBJECT_ID <<< "${line}"

echo "----------------------------------------"
echo "Task ID     : ${SLURM_ARRAY_TASK_ID}"
echo "Study       : ${STUDY}"
echo "Subject ID  : ${SUBJECT_ID}"
echo "Subject dir : ${SUBJECT_DIR}"
echo "BIDS root   : ${BIDS_ROOT}"
echo "Out root    : ${OUT_ROOT}"
echo "Host        : $(hostname)"
echo "Start       : $(date)"
echo "----------------------------------------"

# Activate environment
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

# Thread hygiene
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STUDY_BIDS_ROOT="${BIDS_ROOT}/${STUDY}"
STUDY_OUT_ROOT="${OUT_ROOT}/${STUDY}"

mkdir -p "${STUDY_OUT_ROOT}"

# Adjust the antsxmm invocation here to match your installed CLI.
# Common pattern: point antsxmm at the study BIDS root and restrict to one subject.
#
# Replace this block if your exact command differs.
antsxmm \
  --bids-dir "${STUDY_BIDS_ROOT}" \
  --output-dir "${STUDY_OUT_ROOT}" \
  --participant-label "${SUBJECT_ID#sub-}"

status=$?

echo "End   : $(date)"
echo "Code  : ${status}"

exit "${status}"

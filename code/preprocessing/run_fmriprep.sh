#!/usr/bin/env bash
# Publication configuration, adapted from the study's run_preprocess.sh.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash code/preprocessing/run_fmriprep.sh [--dry-run]
Inputs: PAPER_DATASET/bids, PAPER_INPUTS/participants.tsv, FS_LICENSE.
Output: PAPER_OUTPUT/fmriprep. Participants TSV: subject, complete_assessments.
Resource settings: FMRI_NPROCS=8, FMRI_OMP_THREADS=4, FMRI_MEM_MB=24000,
FMRI_BATCH_SIZE=5. Five participants share one native fMRIPrep scheduler.
EOF
    exit 0
fi
[[ $# == 0 || ( $# == 1 && "$1" == "--dry-run" ) ]] || { echo "Use --help" >&2; exit 2; }
DRY_RUN="${1:-}"
PAPER_DATASET="${PAPER_DATASET:-$REPO/data/dataset}"
PAPER_INPUTS="${PAPER_INPUTS:-$REPO/data/inputs}"
PAPER_OUTPUT="${PAPER_OUTPUT:-$REPO/outputs}"
PAPER_CACHE="${PAPER_CACHE:-$HOME/.cache/narrative-paper}"
FMRI_NPROCS="${FMRI_NPROCS:-8}"
FMRI_OMP_THREADS="${FMRI_OMP_THREADS:-4}"
FMRI_MEM_MB="${FMRI_MEM_MB:-24000}"
FMRI_BATCH_SIZE="${FMRI_BATCH_SIZE:-5}"
for N in "$FMRI_NPROCS" "$FMRI_OMP_THREADS" "$FMRI_MEM_MB" "$FMRI_BATCH_SIZE"; do
    [[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "Resource settings must be positive integers." >&2; exit 2; }
done
: "${FS_LICENSE:?Set FS_LICENSE to your FreeSurfer license.txt}"
[[ -s "$FS_LICENSE" && -d "$PAPER_DATASET/bids" && -s "$PAPER_INPUTS/participants.tsv" ]] || { echo "Missing BIDS, participant list or FreeSurfer license." >&2; exit 1; }
SUBJECT_TEXT=$(python3 - "$PAPER_INPUTS/participants.tsv" <<'PY'
import csv, re, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
subjects = [r['subject'].removeprefix('sub-') for r in rows]
if len(subjects) != 23 or len(set(subjects)) != 23 or any(not re.fullmatch('[A-Za-z0-9]+', s) for s in subjects):
    raise ValueError('Expected 23 unique BIDS subject labels.')
print('\n'.join(subjects))
PY
)
mapfile -t SUBJECTS <<< "$SUBJECT_TEXT"
PAPER_DATASET="$(realpath "$PAPER_DATASET")"
PAPER_OUTPUT="$(realpath -m "$PAPER_OUTPUT")"
PAPER_CACHE="$(realpath -m "$PAPER_CACHE")"
FS_LICENSE="$(realpath "$FS_LICENSE")"
IMAGE="nipreps/fmriprep:25.2.7"
if [[ -z "$DRY_RUN" ]]; then
    mkdir -p "$PAPER_OUTPUT/fmriprep" "$PAPER_CACHE/fmriprep-work/templateflow"
    docker pull "$IMAGE"
fi
for ((I=0; I<${#SUBJECTS[@]}; I+=FMRI_BATCH_SIZE)); do
    BATCH=("${SUBJECTS[@]:I:FMRI_BATCH_SIZE}")
    COMMAND=(docker run --rm --cpus "$FMRI_NPROCS"
        -v "$PAPER_DATASET/bids:/data:ro"
        -v "$PAPER_OUTPUT/fmriprep:/out"
        -v "$PAPER_CACHE/fmriprep-work:/work"
        -v "$PAPER_CACHE/fmriprep-work/templateflow:/templateflow"
        -v "$FS_LICENSE:/license.txt:ro"
        -e TEMPLATEFLOW_HOME=/templateflow -e "OMP_NUM_THREADS=$FMRI_OMP_THREADS"
        -e "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$FMRI_OMP_THREADS"
        "$IMAGE" /data /out participant --participant-label "${BATCH[@]}"
        --task-id listening --subject-anatomical-reference first-lex
        --output-spaces MNI152NLin6Asym:res-2 T1w func
        --level full --submm-recon --return-all-components
        --nprocs "$FMRI_NPROCS" --omp-nthreads "$FMRI_OMP_THREADS" --mem "$FMRI_MEM_MB"
        --slice-time-ref start --ignore fieldmaps --use-syn-sdc error
        --fs-license-file /license.txt --random-seed 2026 --notrack
        -w "/work/batch-${BATCH[0]}")
    if [[ -n "$DRY_RUN" ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; else "${COMMAND[@]}"; fi
done

#!/usr/bin/env bash
# Train one configuration at several seeds, one after another, into per-seed run directories.
#
#   cgfm/scripts/run_seeds.sh <config> <seed>... [-- extra CLI arguments...]
#
# A single seed cannot say whether a difference between two arms is a difference. The MP-20 sweep settled that question
# by declaring any margin under three points a tie, which was a stand-in for a variance estimate rather than one; on
# MPTS-52 the atomwise baseline costs about seven GPU-hours, so the variance can just be measured instead. Three seeds
# per arm is the protocol, and the comparison is made on the mean and the spread.
#
# Each seed writes to <save dir>/seed<n>, so the seed is recoverable from the path. Lightning's automatic version
# numbering is not used for this, because a version number records how many times a command was run and not what it was
# run with. A seed whose directory already holds a last.ckpt is skipped, so an interrupted sweep can be restarted.
#
# Set CGFM_SEED_SAVE_DIR to place the runs somewhere other than the config's own logger directory. Run from the
# repository root.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <config> <seed>... [-- extra CLI arguments...]" >&2
    exit 2
fi

CONFIG="$1"
shift

SEEDS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    SEEDS+=("$1")
    shift
done
[[ $# -gt 0 ]] && shift

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "no seeds given" >&2
    exit 2
fi

PYTHON="${CGFM_PYTHON:-omg}"
SAVE_DIR="${CGFM_SEED_SAVE_DIR:-}"
if [[ -z "${SAVE_DIR}" ]]; then
    SAVE_DIR=$(grep -A2 'CSVLogger' "${CONFIG}" | grep 'save_dir' | head -1 | sed 's/.*save_dir: *"\?\([^"]*\)"\?.*/\1/')
    if [[ -z "${SAVE_DIR}" ]]; then
        echo "could not read save_dir from ${CONFIG}; set CGFM_SEED_SAVE_DIR" >&2
        exit 1
    fi
fi

echo "Training ${CONFIG} at seeds ${SEEDS[*]}, into ${SAVE_DIR}/seed<n>."

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${SAVE_DIR}/seed${SEED}"
    if [[ -s "${OUT_DIR}/checkpoints/last.ckpt" ]]; then
        echo "=== seed ${SEED}: already trained, skipping ==="
        continue
    fi
    echo "=== seed ${SEED} ==="
    mkdir -p "${OUT_DIR}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PYTHON}" fit \
        --config "${CONFIG}" \
        --seed_everything "${SEED}" \
        --trainer.logger.init_args.save_dir "${OUT_DIR}" \
        --trainer.logger.init_args.version "" \
        "$@" 2>&1 | tee "${OUT_DIR}/fit.log"
done

echo "All seeds finished."

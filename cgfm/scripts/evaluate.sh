#!/usr/bin/env bash
# Generate structures for one trained arm and score them on the test split.
#
#   cgfm/scripts/evaluate.sh <arm> <seed> <checkpoint|auto> [extra CLI arguments...]
#
# Passing "auto" picks the best_match_rate checkpoint of that run, wherever Lightning happened to put it.
#
# Draws are generated one per reference composition, repeated CGFM_NUM_DRAWS times with different sampling seeds. The
# first draw gives the one-shot metrics the arms are compared on; all draws together give the best-of-n metrics. The
# model seed is left at the training seed so that only the sampling noise differs between draws.
#
# Results land in <run dir>/eval/nfe<steps>/, so the same run can be evaluated at several numbers of Euler steps
# without overwriting anything. Set CGFM_NFE to change the step count.
#
# Run from the repository root.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <atomwise|kmedoids|shells|learned> <seed> <checkpoint|auto> [extra CLI arguments...]" >&2
    exit 2
fi

ARM="$1"
SEED="$2"
CHECKPOINT="$3"
shift 3

CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
BASE_CONFIG="${CGFM_BASE_CONFIG:-${CONFIG_DIR}/base_mpts52.yaml}"
ARM_CONFIG="${CONFIG_DIR}/arm_${ARM}.yaml"
RUN_DIR="${CGFM_RUN_ROOT:-runs}/${ARM}/seed${SEED}"
NUM_DRAWS="${CGFM_NUM_DRAWS:-5}"
NFE="${CGFM_NFE:-210}"
REFERENCE="${CGFM_REFERENCE:-omg/data/mpts_52/test.lmdb}"
EVAL_DIR="${RUN_DIR}/eval/nfe${NFE}"

if [[ "${CHECKPOINT}" == "auto" ]]; then
    CHECKPOINT=$(find "${RUN_DIR}" -name 'best_match_rate*.ckpt' -print | sort | tail -1)
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "no best_match_rate checkpoint under ${RUN_DIR}" >&2
        exit 1
    fi
fi

mkdir -p "${EVAL_DIR}"

DRAW_FILES=()
for (( DRAW=0; DRAW<NUM_DRAWS; DRAW++ )); do
    XYZ_FILE="${EVAL_DIR}/draw${DRAW}.xyz"
    DRAW_FILES+=("${XYZ_FILE}")
    if [[ -s "${XYZ_FILE}" ]]; then
        echo "Reusing existing ${XYZ_FILE}."
        continue
    fi
    echo "Generating draw ${DRAW} of ${NUM_DRAWS} for arm ${ARM} seed ${SEED} at ${NFE} steps."
    python -m cgfm.main predict \
        --config "${BASE_CONFIG}" \
        --config "${ARM_CONFIG}" \
        --seed_everything "$(( SEED * 1000 + DRAW ))" \
        --ckpt_path "${CHECKPOINT}" \
        --model.si.init_args.integration_time_steps "${NFE}" \
        --model.generation_xyz_filename "${XYZ_FILE}" \
        "$@"
done

echo "Scoring ${NUM_DRAWS} draw(s) against ${REFERENCE}."
python -m cgfm.scripts.score \
    --generated "${DRAW_FILES[@]}" \
    --reference "${REFERENCE}" \
    --out "${EVAL_DIR}/score.json" \
    --label "${ARM}/seed${SEED}/nfe${NFE}" 2>&1 | tee "${EVAL_DIR}/score.log"

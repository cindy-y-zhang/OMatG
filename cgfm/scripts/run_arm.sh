#!/usr/bin/env bash
# Train one arm at one seed.
#
#   cgfm/scripts/run_arm.sh <arm> <seed> [extra CLI arguments...]
#
# where <arm> is one of atomwise, kmedoids, shells, learned. Everything lands under runs/<arm>/seed<seed>/, which is
# what cgfm/scripts/evaluate.sh and cgfm/scripts/collect_results.py expect.
#
# Run from the repository root, because the configuration files use paths relative to it.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <atomwise|kmedoids|shells|learned> <seed> [extra CLI arguments...]" >&2
    exit 2
fi

ARM="$1"
SEED="$2"
shift 2

CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
BASE_CONFIG="${CGFM_BASE_CONFIG:-${CONFIG_DIR}/base_mpts52.yaml}"
ARM_CONFIG="${CONFIG_DIR}/arm_${ARM}.yaml"
RUN_DIR="${CGFM_RUN_ROOT:-runs}/${ARM}/seed${SEED}"

if [[ ! -f "${ARM_CONFIG}" ]]; then
    echo "no configuration for arm '${ARM}' at ${ARM_CONFIG}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"
echo "Training arm ${ARM} at seed ${SEED} into ${RUN_DIR}."

python -m cgfm.main fit \
    --config "${BASE_CONFIG}" \
    --config "${ARM_CONFIG}" \
    --seed_everything "${SEED}" \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.init_args.save_dir "${RUN_DIR}" \
    "$@" 2>&1 | tee -a "${RUN_DIR}/train.log"

echo "Checkpoints:"
find "${RUN_DIR}" -name '*.ckpt' -print

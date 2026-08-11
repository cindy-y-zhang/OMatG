#!/usr/bin/env bash
# Stage 1 go/no-go: does any coarse-to-fine path beat the atomwise baseline?
#
#   cgfm/scripts/sweep_eta.sh [extra CLI arguments...]
#
# Trains the k-medoids arm at eta in {0, 0.25, 0.5, 0.75} on MP-20 at reduced budget, then reports the best validation
# match rate of each. If no eta beats atomwise, the MPTS-52 budget is not spent.
#
# eta = 0 is the atomwise baseline rather than a k-medoids run, because at eta = 0 the grouping has no effect on the
# path at all: the arm factory rejects that combination instead of silently accepting a partition it would ignore.
#
# Runs land in sweeps/<dataset>/eta<value>/. Run from the repository root.

set -euo pipefail

CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
BASE_CONFIG="${CGFM_BASE_CONFIG:-${CONFIG_DIR}/base_mpts52.yaml}"
SWEEP_ROOT="${CGFM_SWEEP_ROOT:-sweeps}"
ETAS="${CGFM_SWEEP_ETAS:-0 0.25 0.5 0.75}"
SEED="${CGFM_SWEEP_SEED:-0}"
WORKERS="${CGFM_PRECOMPUTE_WORKERS:-8}"
WANDB_PROJECT="${CGFM_WANDB_PROJECT:-cgfm-s1}"
WANDB_ENTITY="${CGFM_WANDB_ENTITY:-cindyz}"
DATASET="mp20"
DATA_NAME="mp_20"
GROUP_DIR="cgfm/groups/${DATA_NAME}"

# The sweep only uses the k-medoids partition, but precompute writes both fixed partitions together and the data
# module loads both so the diagnostics can compare against each.
for SPLIT in train val; do
    if [[ ! -f "${GROUP_DIR}/${SPLIT}.kmedoids.npz" ]]; then
        echo "Precomputing ${DATA_NAME} ${SPLIT} partitions."
        python -m cgfm.scripts.precompute_groups \
            --data "omg/data/${DATA_NAME}/${SPLIT}.lmdb" --out-dir "${GROUP_DIR}" --workers "${WORKERS}"
    fi
done

for ETA in ${ETAS}; do
    if [[ "${ETA}" == "0" || "${ETA}" == "0.0" ]]; then
        ARM_CONFIG="${CONFIG_DIR}/arm_atomwise.yaml"
    else
        ARM_CONFIG="${CONFIG_DIR}/arm_kmedoids.yaml"
    fi
    RUN_DIR="${SWEEP_ROOT}/${DATASET}/eta${ETA}"
    mkdir -p "${RUN_DIR}"
    echo "Training ${DATASET} at eta ${ETA} into ${RUN_DIR}."
    LOGGER_CONFIG="[
        {\"class_path\":\"lightning.pytorch.loggers.CSVLogger\",
         \"init_args\":{\"save_dir\":\"${RUN_DIR}\",\"name\":\"\"}},
        {\"class_path\":\"lightning.pytorch.loggers.WandbLogger\",
         \"init_args\":{\"name\":\"mp20-eta${ETA}-seed${SEED}\",
                      \"project\":\"${WANDB_PROJECT}\",\"entity\":\"${WANDB_ENTITY}\",
                      \"save_dir\":\"${RUN_DIR}\",\"log_model\":false}}
    ]"
    # The dataset overlay is applied after the arm overlay, because the arm overlay names the MPTS-52 group files
    # and the sweep has to redirect those to the group files of the dataset being swept.
    python -m cgfm.main fit \
        --config "${BASE_CONFIG}" \
        --config "${ARM_CONFIG}" \
        --config "${CONFIG_DIR}/sweep_${DATASET}.yaml" \
        --seed_everything "${SEED}" \
        --model.si.init_args.eta "${ETA}" \
        --trainer.default_root_dir "${RUN_DIR}" \
        --trainer.logger "${LOGGER_CONFIG}" \
        "$@" 2>&1 | tee -a "${RUN_DIR}/train.log"
done

echo
echo "Sweep finished. Comparing best validation match rate per eta:"
python -m cgfm.scripts.collect_results --runs "${SWEEP_ROOT}" --source validation

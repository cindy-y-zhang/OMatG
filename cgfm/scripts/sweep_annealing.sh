#!/usr/bin/env bash
# Re-tune the inference velocity annealing factor of an already trained arm.
#
#   cgfm/scripts/sweep_annealing.sh <run dir> <atomwise|kmedoids|shells|learned> <eta> [extra CLI arguments...]
#
# The sampler multiplies predicted position velocities by (1 + factor * t), and the factor inherited from OMatG was
# tuned for the atomwise path. With the default bump the within-group schedule has terminal slope 1 + eta, so that
# factor overshoots by the same ratio on the component that carries most of the displacement. The eta sweep therefore
# scored every coarse-grained arm with a mistuned sampler, by a margin that grows with eta. This script asks what each
# arm scores at its own factor, which needs no retraining: the predicted optimum is roughly the tuned value divided by
# (1 + eta).
#
# Only match_rate is meaningful here. Generation never evaluates a grouping, so the group file and, for the learned
# arm, the membership temperature affect the reported val_loss_total but not a single generated structure.
#
# Every factor reuses the same seed, so all runs start from identical x_0 draws and the comparison across factors is
# paired. Results land in <run dir>/anneal/val-<batches>/factor<value>/metrics.csv, which
# cgfm/scripts/collect_annealing.py reads.
#
# Set CGFM_ANNEAL_VAL_BATCHES to score a prefix of the validation split instead of all of it, which is much cheaper and
# enough to locate the optimum. It is part of the output path because match rates on a prefix and on the whole split are
# different quantities, so a cheap sweep must not be mistaken for a confirmation. Run from the repository root.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <run dir> <atomwise|kmedoids|shells|learned> <eta> [extra CLI arguments...]" >&2
    exit 2
fi

RUN_DIR="$1"
ARM="$2"
ETA="$3"
shift 3

PYTHON="${CGFM_PYTHON:-python}"
CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
BASE_CONFIG="${CGFM_BASE_CONFIG:-${CONFIG_DIR}/base_mpts52.yaml}"
DATASET="${CGFM_ANNEAL_DATASET:-mp20}"
DATA_NAME="${CGFM_ANNEAL_DATA_NAME:-mp_20}"
SEED="${CGFM_ANNEAL_SEED:-0}"
# The tuned OMatG value first, so that the reference point is measured before anything else.
FACTORS="${CGFM_ANNEAL_FACTORS:-10.182659004291072 8.0 6.0 4.0}"
# Matching dominates the cost and runs on CPU, so a prefix of the split is the cheap way to locate the optimum. An
# integer is a number of batches and a float is a fraction of the split, following Lightning's own convention.
VAL_BATCHES="${CGFM_ANNEAL_VAL_BATCHES:-1.0}"
SCOPE="${CGFM_ANNEAL_SCOPE:-val-${VAL_BATCHES}}"
NUMBER_CPUS="${CGFM_ANNEAL_CPUS:-$(nproc)}"

CHECKPOINT="${CGFM_ANNEAL_CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT=$(find "${RUN_DIR}" -name 'best_match_rate*.ckpt' -print | sort | tail -1)
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "no best_match_rate checkpoint under ${RUN_DIR}" >&2
        exit 1
    fi
fi

if [[ "${ARM}" == "atomwise" ]]; then
    ARM_CONFIG="${CONFIG_DIR}/arm_atomwise.yaml"
else
    ARM_CONFIG="${CONFIG_DIR}/arm_${ARM}.yaml"
fi

# The dataset overlay names the k-medoids group files, so the coordination-shell arm has to redirect them again here.
GROUP_METHOD="kmedoids"
if [[ "${ARM}" == "shells" ]]; then
    GROUP_METHOD="shells"
fi
GROUP_DIR="cgfm/groups/${DATA_NAME}"

echo "Re-tuning ${ARM} at eta ${ETA} from ${CHECKPOINT}."
echo "Factors: ${FACTORS}. Validation batches: ${VAL_BATCHES}. Matching on ${NUMBER_CPUS} CPUs."

for FACTOR in ${FACTORS}; do
    OUT_DIR="${RUN_DIR}/anneal/${SCOPE}/factor${FACTOR}"
    if [[ -s "${OUT_DIR}/metrics.csv" ]]; then
        echo "=== ${ARM} eta ${ETA} at annealing factor ${FACTOR}: reusing existing result ==="
        continue
    fi
    echo "=== ${ARM} eta ${ETA} at annealing factor ${FACTOR} ==="
    mkdir -p "${OUT_DIR}"
    LOGGER_CONFIG="[
        {\"class_path\":\"lightning.pytorch.loggers.CSVLogger\",
         \"init_args\":{\"save_dir\":\"${OUT_DIR}\",\"name\":\"\",\"version\":\"\"}}
    ]"
    "${PYTHON}" -m cgfm.main validate \
        --config "${BASE_CONFIG}" \
        --config "${ARM_CONFIG}" \
        --config "${CONFIG_DIR}/sweep_${DATASET}.yaml" \
        --seed_everything "${SEED}" \
        --ckpt_path "${CHECKPOINT}" \
        --model.si.init_args.eta "${ETA}" \
        --model.si.init_args.velocity_annealing_factor "${FACTOR}" \
        --model.number_cpus "${NUMBER_CPUS}" \
        --data.train_group_file "${GROUP_DIR}/train.${GROUP_METHOD}.npz" \
        --data.val_group_file "${GROUP_DIR}/val.${GROUP_METHOD}.npz" \
        --trainer.limit_val_batches "${VAL_BATCHES}" \
        --trainer.logger "${LOGGER_CONFIG}" \
        "$@" 2>&1 | tee "${OUT_DIR}/validate.log"
done

echo
echo "Sweep finished. Match rate per annealing factor:"
"${PYTHON}" cgfm/scripts/collect_annealing.py --runs "${RUN_DIR}"

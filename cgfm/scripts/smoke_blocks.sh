#!/usr/bin/env bash
# Tiny end-to-end checks of the rigid-block stack, plus the 100-structure overfit command.
#
#   cgfm/scripts/smoke_blocks.sh
#   cgfm/scripts/smoke_blocks.sh overfit
#   cgfm/scripts/smoke_blocks.sh timing [oracle_coord|joint]
#
# The default path builds a tiny split, precomputes block tables on it, and trains oracle and joint for two epochs at
# five Euler steps. The overfit path is printed, not run: it is the G3 gate that must reach 80 per cent training-set
# match before the nine GPU jobs are submitted.
#
# Run from the repository root.

set -euo pipefail

PYTHON="${CGFM_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON="python"
    elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python3"
    fi
fi
DATA_DIR="cgfm/smoke_data"
BLOCK_DIR="cgfm/smoke_blocks"
RUN_ROOT="cgfm/smoke_runs/blocks"
CONFIGS="cgfm/configs"
SOURCE="${CGFM_DATA_DIR:-omg/data/mpts_52}"
PRODUCTION_BLOCK_DIR="${CGFM_BLOCK_DIR:-cgfm/blocks/mpts_52}"

if [[ "${1:-smoke}" == "timing" ]]; then
    ARM="${2:-oracle_coord}"
    if [[ "${ARM}" == "oracle_coord" ]]; then
        OVERLAY="${CONFIGS}/block_oracle_mpts52.yaml"
    elif [[ "${ARM}" == "joint" ]]; then
        OVERLAY="${CONFIGS}/block_joint_mpts52.yaml"
    else
        echo "usage: $0 timing [oracle_coord|joint]" >&2
        exit 2
    fi
    for artifact in manifest.json templates.pkl train.npz val.npz test.npz; do
        if [[ ! -s "${PRODUCTION_BLOCK_DIR}/${artifact}" ]]; then
            echo "missing ${PRODUCTION_BLOCK_DIR}/${artifact}; run cgfm/scripts/prepare_blocks_mpts52.sh" >&2
            exit 1
        fi
    done

    TIMING_ROOT="${CGFM_TIMING_ROOT:-cgfm/timing_runs/${ARM}/$(date -u +%Y%m%dT%H%M%SZ)}"
    mkdir -p "${TIMING_ROOT}/checkpoints"
    CALLBACKS="[{\"class_path\":\"lightning.pytorch.callbacks.ModelCheckpoint\",\"init_args\":{\"dirpath\":\"${TIMING_ROOT}/checkpoints\",\"save_top_k\":0,\"save_last\":true}}]"
    echo "== timing five full-data training epochs without validation =="
    TRAIN_START=$(date +%s)
    "${PYTHON}" -m cgfm.block_main fit \
        --config "${CONFIGS}/block_mpts52.yaml" \
        --config "${OVERLAY}" \
        --data.train_dataset.init_args.file_path "${SOURCE}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${SOURCE}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${SOURCE}/test.lmdb" \
        --data.block_dir "${PRODUCTION_BLOCK_DIR}" \
        --model.si.init_args.enable_progress_bar false \
        --trainer.max_epochs 5 \
        --trainer.max_time.hours 4 \
        --trainer.limit_val_batches 0 \
        --trainer.callbacks "${CALLBACKS}" \
        --trainer.logger.init_args.save_dir "${TIMING_ROOT}/train" \
        --trainer.logger.init_args.name "" \
        --trainer.logger.init_args.version ""
    TRAIN_SECONDS=$(( $(date +%s) - TRAIN_START ))

    CHECKPOINT="${TIMING_ROOT}/checkpoints/last.ckpt"
    if [[ ! -s "${CHECKPOINT}" ]]; then
        echo "timing fit did not produce ${CHECKPOINT}" >&2
        exit 1
    fi
    echo "== timing one full 210-step validation =="
    VAL_START=$(date +%s)
    "${PYTHON}" -m cgfm.block_main validate \
        --config "${CONFIGS}/block_mpts52.yaml" \
        --config "${OVERLAY}" \
        --ckpt_path "${CHECKPOINT}" \
        --data.train_dataset.init_args.file_path "${SOURCE}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${SOURCE}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${SOURCE}/test.lmdb" \
        --data.block_dir "${PRODUCTION_BLOCK_DIR}" \
        --model.si.init_args.enable_progress_bar false \
        --trainer.logger.init_args.save_dir "${TIMING_ROOT}/validation" \
        --trainer.logger.init_args.name "" \
        --trainer.logger.init_args.version ""
    VAL_SECONDS=$(( $(date +%s) - VAL_START ))

    "${PYTHON}" - "${TIMING_ROOT}/estimate.json" "${TRAIN_SECONDS}" "${VAL_SECONDS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
train_seconds = int(sys.argv[2])
validation_seconds = int(sys.argv[3])
projected_seconds = 80 * train_seconds + 16 * validation_seconds
payload = {
    "training_epochs_timed": 5,
    "training_seconds": train_seconds,
    "validation_runs_timed": 1,
    "validation_seconds": validation_seconds,
    "projected_400_epoch_hours": projected_seconds / 3600.0,
    "release_target_hours": 24.0,
    "max_time_hours": 28.0,
    "passes_release_target": projected_seconds <= 24 * 3600,
}
path.write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
    echo "Timing report written to ${TIMING_ROOT}/estimate.json."
    exit 0
fi

if [[ "${1:-smoke}" == "overfit" ]]; then
    echo "G3 overfit (submit this; require at least 80 per cent training-set match before full jobs):"
    echo "  ${PYTHON} -m cgfm.scripts.make_subset --data ${SOURCE}/train.lmdb --out cgfm/overfit_data/train.lmdb --count 100"
    echo "  ${PYTHON} -m cgfm.scripts.make_subset --data ${SOURCE}/train.lmdb --out cgfm/overfit_data/val.lmdb --count 100"
    echo "  ${PYTHON} -m cgfm.scripts.make_subset --data ${SOURCE}/test.lmdb --out cgfm/overfit_data/test.lmdb --count 16"
    echo "  ${PYTHON} -m cgfm.scripts.precompute_blocks --data-dir cgfm/overfit_data --out-dir cgfm/overfit_blocks --workers 4"
    echo "  ${PYTHON} -m cgfm.block_main fit \\"
    echo "      --config ${CONFIGS}/block_mpts52.yaml \\"
    echo "      --config ${CONFIGS}/block_oracle_mpts52.yaml \\"
    echo "      --data.train_dataset.init_args.file_path cgfm/overfit_data/train.lmdb \\"
    echo "      --data.val_dataset.init_args.file_path cgfm/overfit_data/val.lmdb \\"
    echo "      --data.pred_dataset.init_args.file_path cgfm/overfit_data/test.lmdb \\"
    echo "      --data.block_dir cgfm/overfit_blocks \\"
    echo "      --data.batch_size 16 \\"
    echo "      --model.consensus_weight 1.0 \\"
    echo "      --model.si.init_args.integration_time_steps 50 \\"
    echo "      --trainer.max_epochs 200 \\"
    echo "      --trainer.max_time.hours 4 \\"
    echo "      --trainer.check_val_every_n_epoch 10 \\"
    echo "      --trainer.logger.init_args.save_dir cgfm/overfit_runs/oracle \\"
    echo "      --trainer.logger.init_args.name \"\" \\"
    echo "      --trainer.logger.init_args.version \"\""
    echo "Select consensus_weight on this gate, freeze it in block_mpts52.yaml, then certify the run:"
    echo "  ${PYTHON} -m cgfm.scripts.check_overfit_gate \\"
    echo "      --metrics cgfm/overfit_runs/oracle/metrics.csv \\"
    echo "      --consensus-weight <selected-weight> \\"
    echo "      --stamp cgfm/blocks/mpts_52/phase1_passed.json"
    exit 0
fi

echo "== building a tiny split =="
"${PYTHON}" -m cgfm.scripts.make_subset --data "${SOURCE}/train.lmdb" --out "${DATA_DIR}/train.lmdb" --count 16
"${PYTHON}" -m cgfm.scripts.make_subset --data "${SOURCE}/val.lmdb"   --out "${DATA_DIR}/val.lmdb"   --count 8
"${PYTHON}" -m cgfm.scripts.make_subset --data "${SOURCE}/test.lmdb"  --out "${DATA_DIR}/test.lmdb"  --count 8

echo
echo "== precomputing block tables =="
"${PYTHON}" -m cgfm.scripts.precompute_blocks --data-dir "${DATA_DIR}" --out-dir "${BLOCK_DIR}" --workers 4

echo
echo "== training oracle and joint for two epochs =="
for arm in oracle_coord joint; do
    if [[ "${arm}" == "oracle_coord" ]]; then
        overlay="${CONFIGS}/block_oracle_mpts52.yaml"
    else
        overlay="${CONFIGS}/block_joint_mpts52.yaml"
    fi
    echo "-- ${arm} --"
    rm -rf "${RUN_ROOT}/${arm}"
    mkdir -p "${RUN_ROOT}/${arm}/seed0"
    "${PYTHON}" -m cgfm.block_main fit \
        --config "${CONFIGS}/block_mpts52.yaml" \
        --config "${overlay}" \
        --config "${CONFIGS}/block_smoke.yaml" \
        --trainer.logger.init_args.save_dir "${RUN_ROOT}/${arm}/seed0" \
        --trainer.logger.init_args.version ""
done

echo
echo "== smoke test finished =="
echo "Overfit command: cgfm/scripts/smoke_blocks.sh overfit"

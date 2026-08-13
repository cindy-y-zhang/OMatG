#!/usr/bin/env bash
# Train one MPTS-52 arm at one seed into a deterministic directory, with a 28-hour Lightning guard.
#
#   cgfm/scripts/run_mpts52_block_job.sh <atomwise|oracle_coord|joint> <seed>
#
# A timed-out job is incomplete, not a shorter result. Resume is from last.ckpt when one exists. Run from the
# repository root.

set -euo pipefail

PYTHON="${CGFM_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON="python"
    elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python3"
    fi
fi
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <atomwise|oracle_coord|joint> <seed> [extra CLI arguments...]" >&2
    exit 2
fi

ARM="$1"
SEED="$2"
shift 2

case "${ARM}" in
    atomwise|oracle_coord|joint) ;;
    *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

RUN_ROOT="${CGFM_RUN_ROOT:-runs-blocks/mpts52}"
OUT_DIR="${RUN_ROOT}/${ARM}/seed${SEED}"
CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
PYTHON_ATOMWISE="${CGFM_PYTHON_ATOMWISE:-omg}"
DATA_DIR="${CGFM_DATA_DIR:-omg/data/mpts_52}"
BLOCK_DIR="${CGFM_BLOCK_DIR:-cgfm/blocks/mpts_52}"

mkdir -p "${OUT_DIR}"
RESUME=()
LAST_CKPT=$("${PYTHON}" - "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [path for path in root.rglob("last*.ckpt") if path.is_file() and path.stat().st_size > 0]
if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
)
if [[ -n "${LAST_CKPT}" ]]; then
    RESUME=(--ckpt_path "${LAST_CKPT}")
    echo "Resuming ${ARM} seed ${SEED} from ${LAST_CKPT}"
fi

cp_if_exists() { [[ -f "$1" ]] && cp "$1" "${OUT_DIR}/$(basename "$1")"; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== ${ARM} seed ${SEED} -> ${OUT_DIR} ==="

if [[ "${ARM}" == "atomwise" ]]; then
    CONFIG="${CONFIG_DIR}/atomwise_mpts52.yaml"
    cp_if_exists "${CONFIG}"
    "${PYTHON_ATOMWISE}" fit \
        --config "${CONFIG}" \
        --seed_everything "${SEED}" \
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
        --trainer.logger.init_args.save_dir "${OUT_DIR}" \
        --trainer.logger.init_args.version "" \
        --trainer.max_time.hours 28 \
        "${RESUME[@]}" \
        "$@" 2>&1 | tee -a "${OUT_DIR}/fit.log"
else
    if [[ "${ARM}" == "oracle_coord" ]]; then
        ARM_CONFIG="${CONFIG_DIR}/block_oracle_mpts52.yaml"
    else
        ARM_CONFIG="${CONFIG_DIR}/block_joint_mpts52.yaml"
    fi
    cp_if_exists "${CONFIG_DIR}/block_mpts52.yaml"
    cp_if_exists "${ARM_CONFIG}"
    [[ -f "${BLOCK_DIR}/manifest.json" ]] && cp "${BLOCK_DIR}/manifest.json" "${OUT_DIR}/block_manifest.json"
    "${PYTHON}" -m cgfm.block_main fit \
        --config "${CONFIG_DIR}/block_mpts52.yaml" \
        --config "${ARM_CONFIG}" \
        --seed_everything "${SEED}" \
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
        --data.block_dir "${BLOCK_DIR}" \
        --trainer.logger.init_args.save_dir "${OUT_DIR}" \
        --trainer.logger.init_args.version "" \
        --trainer.max_time.hours 28 \
        "${RESUME[@]}" \
        "$@" 2>&1 | tee -a "${OUT_DIR}/fit.log"
fi

"${PYTHON}" -m cgfm.scripts.check_training_complete --run-dir "${OUT_DIR}" --expected-epochs 400
echo "Finished ${ARM} seed ${SEED}."

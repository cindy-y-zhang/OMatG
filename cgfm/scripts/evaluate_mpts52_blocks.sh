#!/usr/bin/env bash
# Evaluate one trained MPTS-52 block arm: tune translation and rotation annealing on a validation prefix, lock both,
# select the checkpoint on full validation, and score the test split once.
#
#   cgfm/scripts/evaluate_mpts52_blocks.sh <atomwise|oracle_coord|joint> <seed> [checkpoint|auto]
#
# Atomwise has no rotation field, so only translation annealing is swept. Run from the repository root.

set -euo pipefail

PYTHON="${CGFM_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON="python"
    elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python3"
    fi
fi
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <atomwise|oracle_coord|joint> <seed> [checkpoint|auto] [extra CLI arguments...]" >&2
    exit 2
fi

ARM="$1"
SEED="$2"
CHECKPOINT="${3:-auto}"
REQUESTED_CHECKPOINT="${CHECKPOINT}"
if [[ $# -ge 3 ]]; then
    shift 3
else
    shift 2
fi

RUN_ROOT="${CGFM_RUN_ROOT:-runs-blocks/mpts52}"
RUN_DIR="${RUN_ROOT}/${ARM}/seed${SEED}"
CONFIG_DIR="${CGFM_CONFIG_DIR:-cgfm/configs}"
NFE="${CGFM_NFE:-210}"
VAL_BATCHES="${CGFM_ANNEAL_VAL_BATCHES:-0.2}"
POS_FACTORS="${CGFM_POS_FACTORS:-10.182659004291072 8.0 6.0 4.0 13.0 16.0 20.0}"
ROT_FACTORS="${CGFM_ROT_FACTORS:-10.182659004291072 8.0 6.0 4.0 13.0 16.0 20.0}"
NUMBER_CPUS="${CGFM_ANNEAL_CPUS:-$(nproc)}"
REFERENCE="${CGFM_REFERENCE:-omg/data/mpts_52/test.lmdb}"
DATA_DIR="${CGFM_DATA_DIR:-omg/data/mpts_52}"
BLOCK_DIR="${CGFM_BLOCK_DIR:-cgfm/blocks/mpts_52}"

"${PYTHON}" -m cgfm.scripts.check_training_complete --run-dir "${RUN_DIR}" --expected-epochs 400

if [[ "${CHECKPOINT}" == "auto" ]]; then
    CHECKPOINT=$("${PYTHON}" - "${RUN_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = sorted(path for path in root.rglob("best_match_rate*.ckpt") if path.is_file())
if not candidates:
    candidates = sorted(path for path in root.rglob("last*.ckpt") if path.is_file())
if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
)
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "no training checkpoint under ${RUN_DIR}" >&2
        exit 1
    fi
fi

best_factor() {
    local root="$1"
    "${PYTHON}" - "$root" <<'PY'
from pathlib import Path
import csv, sys
root = Path(sys.argv[1])
best_factor, best_rate = None, None
for metrics in root.glob("factor*/metrics.csv"):
    rate = None
    with metrics.open() as handle:
        for row in csv.DictReader(handle):
            raw = row.get("match_rate", "")
            if raw not in ("", None):
                rate = float(raw)
    if rate is None:
        continue
    factor = float(metrics.parent.name[len("factor"):])
    if best_rate is None or rate > best_rate:
        best_factor, best_rate = factor, rate
if best_factor is None:
    raise SystemExit(f"no match_rate under {root}")
print(best_factor)
PY
}

validate_cmd() {
    local out_dir="$1"
    shift
    mkdir -p "${out_dir}"
    local logger="[{\"class_path\":\"lightning.pytorch.loggers.CSVLogger\",\"init_args\":{\"save_dir\":\"${out_dir}\",\"name\":\"\",\"version\":\"\"}}]"
    if [[ "${ARM}" == "atomwise" ]]; then
        omg validate \
            --config "${CONFIG_DIR}/atomwise_mpts52.yaml" \
            --seed_everything "${SEED}" \
            --ckpt_path "${CHECKPOINT}" \
            --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
            --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
            --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
            --model.si.init_args.integration_time_steps "${NFE}" \
            --model.number_cpus "${NUMBER_CPUS}" \
            --trainer.logger "${logger}" \
            "$@"
    else
        local overlay="${CONFIG_DIR}/block_oracle_mpts52.yaml"
        [[ "${ARM}" == "joint" ]] && overlay="${CONFIG_DIR}/block_joint_mpts52.yaml"
        "${PYTHON}" -m cgfm.block_main validate \
            --config "${CONFIG_DIR}/block_mpts52.yaml" \
            --config "${overlay}" \
            --seed_everything "${SEED}" \
            --ckpt_path "${CHECKPOINT}" \
            --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
            --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
            --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
            --data.block_dir "${BLOCK_DIR}" \
            --model.si.init_args.integration_time_steps "${NFE}" \
            --model.number_cpus "${NUMBER_CPUS}" \
            --trainer.logger "${logger}" \
            "$@"
    fi
}

echo "Sweeping translation annealing for ${ARM} seed ${SEED}."
POS_ROOT="${RUN_DIR}/anneal/pos-val-${VAL_BATCHES}"
for factor in ${POS_FACTORS}; do
    out="${POS_ROOT}/factor${factor}"
    if [[ -s "${out}/metrics.csv" ]]; then
        echo "reusing ${out}"
        continue
    fi
    if [[ "${ARM}" == "atomwise" ]]; then
        validate_cmd "${out}" \
            --model.si.init_args.stochastic_interpolants.1.init_args.velocity_annealing_factor "${factor}" \
            --trainer.limit_val_batches "${VAL_BATCHES}"
    else
        validate_cmd "${out}" \
            --model.si.init_args.pos_annealing_factor "${factor}" \
            --trainer.limit_val_batches "${VAL_BATCHES}"
    fi
done
POS_LOCKED="$(best_factor "${POS_ROOT}")"
echo "Locked translation annealing factor ${POS_LOCKED}."

ROT_LOCKED="${POS_LOCKED}"
if [[ "${ARM}" != "atomwise" ]]; then
    echo "Sweeping rotation annealing with translation locked."
    ROT_ROOT="${RUN_DIR}/anneal/rot-val-${VAL_BATCHES}"
    for factor in ${ROT_FACTORS}; do
        out="${ROT_ROOT}/factor${factor}"
        if [[ -s "${out}/metrics.csv" ]]; then
            echo "reusing ${out}"
            continue
        fi
        validate_cmd "${out}" \
            --model.si.init_args.pos_annealing_factor "${POS_LOCKED}" \
            --model.si.init_args.rot_annealing_factor "${factor}" \
            --trainer.limit_val_batches "${VAL_BATCHES}"
    done
    ROT_LOCKED="$(best_factor "${ROT_ROOT}")"
    echo "Locked rotation annealing factor ${ROT_LOCKED}."
fi

echo "Selecting the checkpoint on full validation at locked annealing."
mapfile -t CHECKPOINT_CANDIDATES < <("${PYTHON}" - "${RUN_DIR}" "${REQUESTED_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
requested = sys.argv[2]
if requested != "auto":
    candidates = [Path(requested)]
else:
    candidates = sorted({
        path.resolve()
        for pattern in ("best_match_rate*.ckpt", "last*.ckpt")
        for path in root.rglob(pattern)
        if path.is_file()
    })
for path in candidates:
    print(path)
PY
)
if [[ ${#CHECKPOINT_CANDIDATES[@]} -eq 0 ]]; then
    echo "no checkpoints available for full-validation selection" >&2
    exit 1
fi

SELECTION_ROOT="${RUN_DIR}/anneal/val-full/pos${POS_LOCKED}_rot${ROT_LOCKED}"
for index in "${!CHECKPOINT_CANDIDATES[@]}"; do
    candidate="${CHECKPOINT_CANDIDATES[${index}]}"
    candidate_id=$("${PYTHON}" - "${candidate}" <<'PY'
import hashlib
import sys
print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])
PY
)
    out="${SELECTION_ROOT}/checkpoint-${candidate_id}"
    mkdir -p "${out}"
    if [[ -s "${out}/metrics.csv" && -s "${out}/checkpoint.txt" && "$( < "${out}/checkpoint.txt" )" == "${candidate}" ]]; then
        echo "reusing full validation for ${candidate}"
        continue
    fi
    printf '%s' "${candidate}" > "${out}/checkpoint.txt"
    CHECKPOINT="${candidate}"
    if [[ "${ARM}" == "atomwise" ]]; then
        validate_cmd "${out}" \
            --model.si.init_args.stochastic_interpolants.1.init_args.velocity_annealing_factor "${POS_LOCKED}" \
            --trainer.limit_val_batches 1.0
    else
        validate_cmd "${out}" \
            --model.si.init_args.pos_annealing_factor "${POS_LOCKED}" \
            --model.si.init_args.rot_annealing_factor "${ROT_LOCKED}" \
            --trainer.limit_val_batches 1.0
    fi
done
CHECKPOINT=$("${PYTHON}" - "${SELECTION_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

best = None
for metrics in Path(sys.argv[1]).glob("checkpoint-*/metrics.csv"):
    rates = []
    with metrics.open() as handle:
        for row in csv.DictReader(handle):
            raw = row.get("match_rate", "")
            if raw not in ("", None):
                rates.append(float(raw))
    marker = metrics.parent / "checkpoint.txt"
    if rates and marker.is_file():
        candidate = (max(rates), marker.read_text())
        if best is None or candidate[0] > best[0]:
            best = candidate
if best is None:
    raise SystemExit(f"no full-validation match_rate under {sys.argv[1]}")
print(best[1])
PY
)
echo "Selected checkpoint ${CHECKPOINT} on full validation."

EVAL_DIR="${RUN_DIR}/eval/nfe${NFE}"
mkdir -p "${EVAL_DIR}"
XYZ_FILE="${EVAL_DIR}/draw0.xyz"
if [[ ! -s "${XYZ_FILE}" ]]; then
    echo "Generating the test split once."
    if [[ "${ARM}" == "atomwise" ]]; then
        omg predict \
            --config "${CONFIG_DIR}/atomwise_mpts52.yaml" \
            --seed_everything "${SEED}" \
            --ckpt_path "${CHECKPOINT}" \
            --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
            --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
            --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
            --model.si.init_args.integration_time_steps "${NFE}" \
            --model.si.init_args.stochastic_interpolants.1.init_args.velocity_annealing_factor "${POS_LOCKED}" \
            --model.generation_xyz_filename "${XYZ_FILE}"
    else
        overlay="${CONFIG_DIR}/block_oracle_mpts52.yaml"
        [[ "${ARM}" == "joint" ]] && overlay="${CONFIG_DIR}/block_joint_mpts52.yaml"
        "${PYTHON}" -m cgfm.block_main predict \
            --config "${CONFIG_DIR}/block_mpts52.yaml" \
            --config "${overlay}" \
            --seed_everything "${SEED}" \
            --ckpt_path "${CHECKPOINT}" \
            --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
            --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
            --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
            --data.block_dir "${BLOCK_DIR}" \
            --model.si.init_args.integration_time_steps "${NFE}" \
            --model.si.init_args.pos_annealing_factor "${POS_LOCKED}" \
            --model.si.init_args.rot_annealing_factor "${ROT_LOCKED}" \
            --model.generation_xyz_filename "${XYZ_FILE}"
    fi
fi

"${PYTHON}" -m cgfm.scripts.score \
    --generated "${XYZ_FILE}" \
    --reference "${REFERENCE}" \
    --out "${EVAL_DIR}/score.json" \
    --label "${ARM}/seed${SEED}/nfe${NFE}" 2>&1 | tee "${EVAL_DIR}/score.log"

echo "Evaluation finished. Locked pos=${POS_LOCKED} rot=${ROT_LOCKED}."

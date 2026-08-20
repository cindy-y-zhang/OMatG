#!/usr/bin/env bash
#
# Generate repeated sampling draws for one finished run and score each of them separately.
#
#   bash direct_geometry/scripts/evaluate.sh <run dir>            # validation at 210, then test at 210 and 50
#   bash direct_geometry/scripts/evaluate.sh <run dir> val 210    # one split at one budget
#
# WHAT IS MEASURED, AND WHY IN THIS ORDER
#
# Validation first, at the production sampling budget, repeated DG_DRAWS times from the checkpoint the run itself selected.
# That gives the spread of one checkpoint's match rate over sampling noise, which is what says whether a reported number is
# worth the digits it is printed with. It is measured on validation because validation is where selection already happened,
# so nothing is spent and nothing is leaked.
#
# Then the test split, at 210 and at 50 steps. The architecture and the checkpoint were both chosen on validation and
# nothing is retuned here: the two budgets are reported side by side rather than the better of the two being quoted, and the
# 50-step number is included precisely because a method that only works at a large sampling budget is a different claim
# from one that works cheaply.
#
# THE ARM IS READ FROM THE RUN, NOT RECOMPUTED
#
# A checkpoint carries weights, not the class that consumes them, so the encoder has to be rebuilt before the weights can
# be loaded. Its two switches are read out of the COMMAND file the launcher wrote beside the run, so an evaluation cannot
# quietly rebuild a different arm than the one that was trained. Recomputing them from the directory name would work until
# somebody renamed a directory.
#
# Draws already on disk are reused, so an interrupted evaluation resumes instead of regenerating.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN_DIR="${1:-}"
ONLY_SPLIT="${2:-}"
ONLY_NFE="${3:-}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "usage: $0 <run dir> [val|test] [steps]" >&2
    exit 2
fi

if [[ -n "${DG_PYTHON:-}" ]]; then PYTHON="${DG_PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
else PYTHON="python3"
fi

BASE_CONFIG="${DG_BASE_CONFIG:-cgfm/configs/atomwise_mpts52.yaml}"
CONFIGS="direct_geometry/configs"
DATA_DIR="${DG_DATA_DIR:-omg/data/mpts_52}"
DRAWS="${DG_DRAWS:-5}"
GPU="${DG_GPU:-0}"
WORKERS="${DG_EVAL_WORKERS:-8}"
BATCH="${DG_EVAL_BATCH:-128}"
SEED="$(basename "${RUN_DIR}" | sed 's/.*_seed//')"

CHECKPOINT="$(find "${RUN_DIR}" -name 'best_match_rate*.ckpt' -print | sort | tail -1)"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "no best_match_rate checkpoint under ${RUN_DIR}; nothing selected it, so there is nothing to evaluate" >&2
    exit 1
fi

COMMAND_FILE="${RUN_DIR}/COMMAND"
if [[ ! -s "${COMMAND_FILE}" ]]; then
    echo "no COMMAND file in ${RUN_DIR}, so the arm this checkpoint belongs to cannot be established" >&2
    exit 1
fi

# The value that followed a flag in the recorded invocation.
recorded() {
    "${PYTHON}" - "${COMMAND_FILE}" "$1" <<'PY'
import sys

lines = [line.rstrip("\n") for line in open(sys.argv[1])]
flag = sys.argv[2]
if flag not in lines or lines.index(flag) + 1 >= len(lines):
    raise SystemExit(f"{sys.argv[1]} records no value for {flag}, so the arm cannot be rebuilt from it")
print(lines[lines.index(flag) + 1])
PY
}

GRAPH="$(recorded --model.model.init_args.encoder.init_args.message_graph)"
FEATURES="$(recorded --model.model.init_args.encoder.init_args.feature_mode)"
echo "$(date -u +%H:%M) $(basename "${RUN_DIR}"): graph ${GRAPH}, features ${FEATURES}, checkpoint ${CHECKPOINT}"

# split:steps pairs. Validation at the production budget only: its job is the sampler's spread, and measuring that at a
# budget nothing is reported at would answer a question nobody asked.
PLAN=("val:210" "test:210" "test:50")
if [[ -n "${ONLY_SPLIT}" ]]; then
    [[ -n "${ONLY_NFE}" ]] || { echo "pass a step count with a split" >&2; exit 2; }
    PLAN=("${ONLY_SPLIT}:${ONLY_NFE}")
fi

for entry in "${PLAN[@]}"; do
    IFS=':' read -r split nfe <<< "${entry}"
    eval_dir="${RUN_DIR}/eval/${split}_nfe${nfe}"
    reference="${DATA_DIR}/${split}.lmdb"
    [[ -s "${reference}" ]] || { echo "missing ${reference}" >&2; exit 1; }
    mkdir -p "${eval_dir}"

    if [[ -s "${eval_dir}/SCORE.json" ]]; then
        echo "$(date -u +%H:%M)   ${split} at ${nfe} steps already scored"
        continue
    fi

    draw_files=()
    for (( draw=0; draw<DRAWS; draw++ )); do
        xyz="${eval_dir}/draw${draw}.xyz"
        draw_files+=("${xyz}")
        if [[ -s "${xyz}" ]]; then
            echo "$(date -u +%H:%M)   reusing ${xyz}"
            continue
        fi
        echo "$(date -u +%H:%M)   ${split} draw ${draw} of ${DRAWS} at ${nfe} steps"
        # The model seed stays at the training seed and only the draw index moves it, so what differs between draws is
        # the sampling noise and not the weights.
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m omg.main predict \
            --config "${BASE_CONFIG}" \
            --config "${CONFIGS}/encoder.yaml" \
            --model.model.init_args.encoder.init_args.message_graph "${GRAPH}" \
            --model.model.init_args.encoder.init_args.feature_mode "${FEATURES}" \
            --seed_everything "$(( SEED * 1000 + draw ))" \
            --ckpt_path "${CHECKPOINT}" \
            --data.pred_dataset.init_args.file_path "${reference}" \
            --data.batch_size "${BATCH}" \
            --model.si.init_args.integration_time_steps "${nfe}" \
            --model.generation_xyz_filename "${xyz}" \
            --model.number_cpus "${WORKERS}" \
            --trainer.logger false >> "${eval_dir}/generate.log" 2>&1
    done

    "${PYTHON}" -m direct_geometry.scripts.evaluate \
        --generated "${draw_files[@]}" \
        --reference "${reference}" \
        --split "${split}" \
        --integration-time-steps "${nfe}" \
        --checkpoint "${CHECKPOINT}" \
        --workers "${WORKERS}" \
        --out "${eval_dir}/SCORE.json" 2>&1 | tee "${eval_dir}/score.log"
done

echo "$(date -u +%H:%M) $(basename "${RUN_DIR}"): evaluation finished"

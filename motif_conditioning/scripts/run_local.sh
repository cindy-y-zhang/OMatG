#!/usr/bin/env bash
#
# Staged local protocol for global motif-census conditioning.
#
#   bash motif_conditioning/scripts/run_local.sh screen      # stage 1: M@K=32 against D
#   bash motif_conditioning/scripts/run_local.sh content     # stage 2: X@K=32
#   bash motif_conditioning/scripts/run_local.sh dose        # stage 3: K = 2, 8
#   bash motif_conditioning/scripts/run_local.sh guidance 32 5,20,100   # stage 4, eval only
#   bash motif_conditioning/scripts/run_local.sh report
#
# Stages are separate commands rather than one pipeline because the preregistration has a
# stop rule after stage 1, and a pipeline that runs past its own stop rule is not a
# protocol. D is not retrained here: motif_conditioning/reports/PREREGISTRATION.json records
# why the existing 2000-step D run is the correct paired baseline.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:-screen}"
case "${MODE}" in
    screen|content|dose|guidance|report) ;;
    *) echo "usage: $0 {screen|content|dose|guidance <K>|report}" >&2; exit 2 ;;
esac

PYTHON="${MC_PYTHON:-.venv/bin/python}"
GPU="${MC_GPU:-0}"
DATA_DIR="${MC_DATA_DIR:-omg/data/mpts_52}"
ARTIFACT_ROOT="${MC_ARTIFACT_ROOT:-motif_conditioning/artifacts/mpts_52}"
RUN_ROOT="${MC_RUN_ROOT:-motif_conditioning/runs}"
REPORT_ROOT="${MC_REPORT_ROOT:-motif_conditioning/reports}"
STEPS="${MC_STEPS:-2000}"
SEED="${MC_SEED:-0}"
BASELINE="${MC_BASELINE:-joint_geometry/runs/budget2000/oracle/D_seed0/eval/draw0/OUTCOMES.json}"

BASE="cgfm/configs/atomwise_mpts52.yaml"
DIRECT="direct_geometry/configs/encoder.yaml"
CENSUS="motif_conditioning/configs/census.yaml"
CONFIGS="motif_conditioning/configs"

mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

latest_checkpoint() {
    local run="$1"
    local candidates=("${run}"/checkpoints/last*.ckpt)
    if [[ ! -e "${candidates[0]}" ]]; then
        echo "no final checkpoint under ${run}/checkpoints" >&2
        return 1
    fi
    printf '%s\n' "${candidates[-1]}"
}

census_fit() {
    local arm="$1" prototypes="$2"
    local run="${RUN_ROOT}/${arm}_k${prototypes}_seed${SEED}"
    if [[ -e "${run}/COMPLETE" ]]; then
        echo "Using completed ${run}."
        return
    fi
    if [[ -e "${run}" ]]; then
        echo "${run} exists without COMPLETE; move it aside before retrying." >&2
        exit 1
    fi
    mkdir -p "${run}"
    local command=(
        "${PYTHON}" -m motif_conditioning.main fit
        --config "${BASE}"
        --config "${DIRECT}"
        --config "${CENSUS}"
        --config "${CONFIGS}/${arm}.yaml"
        --config direct_geometry/configs/local.yaml
        --config joint_geometry/configs/screen_train.yaml
        --seed_everything "${SEED}"
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb"
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb"
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb"
        --data.census_dir "${ARTIFACT_ROOT}/motif${prototypes}"
        --model.sampler.init_args.census_dimension "${prototypes}"
        --model.model.init_args.encoder.init_args.census_dimension "${prototypes}"
        --trainer.max_steps "${STEPS}"
        --trainer.max_epochs 1000
        --trainer.logger.init_args.save_dir "${run}"
        --trainer.logger.init_args.name ""
        --trainer.logger.init_args.version ""
    )
    printf '%q ' "${command[@]}" > "${run}/command.txt"
    printf '\n' >> "${run}/command.txt"
    CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}" 2>&1 | tee "${run}/train.log"
    latest_checkpoint "${run}" >/dev/null
    touch "${run}/COMPLETE"
}

census_evaluate() {
    local arm="$1" prototypes="$2" guidance="${3:-1.0}"
    local run="${RUN_ROOT}/${arm}_k${prototypes}_seed${SEED}"
    local checkpoint
    checkpoint="$(latest_checkpoint "${run}")"
    local evaluate="${run}/eval/w${guidance}"
    if [[ -e "${evaluate}/OUTCOMES.json" ]]; then return; fi
    mkdir -p "${evaluate}"
    JG_INFERENCE_DRAW=0 CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m motif_conditioning.main validate \
        --config "${BASE}" \
        --config "${DIRECT}" \
        --config "${CENSUS}" \
        --config "${CONFIGS}/${arm}.yaml" \
        --config joint_geometry/configs/evaluate_local.yaml \
        --seed_everything "${SEED}" \
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
        --data.census_dir "${ARTIFACT_ROOT}/motif${prototypes}" \
        --model.sampler.init_args.census_dimension "${prototypes}" \
        --model.model.init_args.encoder.init_args.census_dimension "${prototypes}" \
        --model.model.init_args.encoder.init_args.guidance_scale "${guidance}" \
        --model.store_validation_structures_path "${evaluate}/structures.xyz" \
        --trainer.logger.init_args.save_dir "${evaluate}" \
        --trainer.logger.init_args.name "" \
        --trainer.logger.init_args.version "" \
        --ckpt_path "${checkpoint}" 2>&1 | tee "${evaluate}/validate.log"
    score_saved_validation "${arm}_k${prototypes}_w${guidance}" "${evaluate}"
}

score_saved_validation() {
    local label="$1" evaluate="$2"
    local generated="" reference="" path
    for path in "${evaluate}"/structures_epoch_*_step_*.xyz; do
        if [[ "${path}" != *_ref.xyz ]]; then generated="${path}"; fi
    done
    if [[ -n "${generated}" ]]; then reference="${generated%.xyz}_ref.xyz"; fi
    if [[ -z "${generated}" || ! -e "${reference}" ]]; then
        echo "validation structures were not written under ${evaluate}" >&2
        exit 1
    fi
    "${PYTHON}" -m joint_geometry.scripts.score_outcomes \
        --generated "${generated}" \
        --reference "${reference}" \
        --out "${evaluate}/OUTCOMES.json" \
        --arm "${label}" \
        --seed "${SEED}" \
        --draw 0
}

report() {
    "${PYTHON}" -m motif_conditioning.scripts.census_report \
        --run-root "${RUN_ROOT}" \
        --baseline "${BASELINE}" \
        --precheck "${REPORT_ROOT}/CENSUS-PRECHECK.json" \
        --out "${REPORT_ROOT}/CENSUS-GATE.json"
}

case "${MODE}" in
    screen)
        census_fit M 32
        census_evaluate M 32 1.0
        report
        ;;
    content)
        census_fit X 32
        census_evaluate X 32 1.0
        report
        ;;
    dose)
        for prototypes in 2 8; do
            census_fit M "${prototypes}"
            census_evaluate M "${prototypes}" 1.0
        done
        report
        ;;
    guidance)
        PROTOTYPES="${2:-32}"
        IFS=',' read -r -a SCALES <<< "${3:-0.0,0.5,1.5,2.0,3.0}"
        for scale in "${SCALES[@]}"; do
            census_evaluate M "${PROTOTYPES}" "${scale}"
        done
        report
        ;;
    report)
        report
        ;;
esac

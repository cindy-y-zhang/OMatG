#!/usr/bin/env bash
#
# Staged RTX PRO 6000 protocol for the joint geometry state.
#
#   bash joint_geometry/scripts/run_local.sh all
#   bash joint_geometry/scripts/run_local.sh smoke
#   bash joint_geometry/scripts/run_local.sh calibrate
#   bash joint_geometry/scripts/run_local.sh oracle
#   bash joint_geometry/scripts/run_local.sh screen
#   bash joint_geometry/scripts/run_local.sh report
#   bash joint_geometry/scripts/run_local.sh arm H
#
# The ``arm`` mode trains and evaluates one named arm at the oracle stage's
# budget, so a control the sequential protocol skipped can be added later
# without rerunning the arms already on disk or restating their commands.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:-all}"
case "${MODE}" in
    all|smoke|calibrate|oracle|screen|report) ;;
    arm)
        SINGLE_ARM="${2:-}"
        case "${SINGLE_ARM}" in
            D|R) ;;
            S|H|O|P|J) ;;
            *) echo "usage: $0 arm {S|D|R|H|O|P|J}" >&2; exit 2 ;;
        esac
        ;;
    *) echo "usage: $0 {all|smoke|calibrate|oracle|screen|report|arm <ARM>}" >&2; exit 2 ;;
esac

PYTHON="${JG_PYTHON:-.venv/bin/python}"
GPU="${JG_GPU:-0}"
DATA_DIR="${JG_DATA_DIR:-omg/data/mpts_52}"
SMOKE_DATA_DIR="${JG_SMOKE_DATA_DIR:-cgfm/smoke_data}"
MEMORIZE_DATA_DIR="${JG_MEMORIZE_DATA_DIR:-cgfm/overfit_data}"
ARTIFACT_ROOT="${JG_ARTIFACT_ROOT:-joint_geometry/artifacts/mpts_52}"
ARTIFACT_OVERRIDE="${JG_ARTIFACT:-}"
ARTIFACT="${ARTIFACT_OVERRIDE:-${ARTIFACT_ROOT}/cn-rdf4}"
DIMENSION_OVERRIDE="${JG_GEOMETRY_DIMENSION:-}"
GEOMETRY_DIMENSION="${DIMENSION_OVERRIDE:-4}"
MESSAGE_GRAPH="${JG_MESSAGE_GRAPH:-periodic_distance}"
SMOKE_ARTIFACT_OVERRIDE="${JG_SMOKE_ARTIFACT:-}"
MEMORIZE_ARTIFACT_OVERRIDE="${JG_MEMORIZE_ARTIFACT:-}"
RUN_ROOT="${JG_RUN_ROOT:-joint_geometry/runs}"
REPORT_ROOT="${JG_REPORT_ROOT:-joint_geometry/reports}"
RETRY_INCOMPLETE="${JG_RETRY_INCOMPLETE:-0}"
LOCAL_STEPS="${JG_LOCAL_STEPS:-2000}"
ORACLE_STEPS="${JG_ORACLE_STEPS:-200}"
IFS=',' read -r -a SEEDS <<< "${JG_SEEDS:-0,1,2}"
RUN_R="${JG_RUN_R:-1}"
BASE="cgfm/configs/atomwise_mpts52.yaml"
DIRECT="direct_geometry/configs/encoder.yaml"
JOINT="joint_geometry/configs/mpts52.yaml"
CONFIGS="joint_geometry/configs"

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

joint_fit() {
    local arm="$1" seed="$2" steps="$3" weight="$4" mode="$5"
    local run="${RUN_ROOT}/${mode}/${arm}_seed${seed}"
    if [[ -e "${run}/COMPLETE" ]]; then
        echo "Using completed ${run}."
        return
    fi
    if [[ -e "${run}" ]]; then
        if [[ "${RETRY_INCOMPLETE}" != 1 ]]; then
            echo "${run} exists without COMPLETE; move it aside or set JG_RETRY_INCOMPLETE=1." >&2
            exit 1
        fi
        echo "Retrying incomplete ${run}; deterministic outputs will be overwritten."
    fi
    mkdir -p "${run}"
    local command=(
        "${PYTHON}" -m joint_geometry.main fit
        --config "${JOINT}"
        --config "${CONFIGS}/${arm}.yaml"
        --config "${CONFIGS}/local.yaml"
        --config "${CONFIGS}/screen_train.yaml"
        --seed_everything "${seed}"
        --model.geometry_weight "${weight}"
        --model.sampler.init_args.geometry_dimension "${GEOMETRY_DIMENSION}"
        --model.model.init_args.encoder.init_args.geometry_dimension "${GEOMETRY_DIMENSION}"
        --model.model.init_args.encoder.init_args.message_graph "${MESSAGE_GRAPH}"
        --data.geometry_dir "${ARTIFACT}"
        --trainer.max_steps "${steps}"
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

direct_fit() {
    local arm="$1" seed="$2" steps="$3" mode="$4"
    local run="${RUN_ROOT}/${mode}/${arm}_seed${seed}"
    if [[ -e "${run}/COMPLETE" ]]; then
        echo "Using completed ${run}."
        return
    fi
    if [[ -e "${run}" ]]; then
        if [[ "${RETRY_INCOMPLETE}" != 1 ]]; then
            echo "${run} exists without COMPLETE; move it aside or set JG_RETRY_INCOMPLETE=1." >&2
            exit 1
        fi
        echo "Retrying incomplete ${run}; deterministic outputs will be overwritten."
    fi
    mkdir -p "${run}"
    local command=(
        "${PYTHON}" -m omg.main fit
        --config "${BASE}"
        --config "${DIRECT}"
        --config "${CONFIGS}/${arm}.yaml"
        --config direct_geometry/configs/local.yaml
        --config "${CONFIGS}/screen_train.yaml"
        --seed_everything "${seed}"
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb"
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb"
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb"
        --model.model.init_args.encoder.init_args.message_graph "${MESSAGE_GRAPH}"
        --trainer.max_steps "${steps}"
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

evaluate_joint() {
    local arm="$1" seed="$2" mode="$3" weight="$4" draw="${5:-0}"
    local run="${RUN_ROOT}/${mode}/${arm}_seed${seed}"
    local checkpoint
    checkpoint="$(latest_checkpoint "${run}")"
    local evaluate="${run}/eval/draw${draw}"
    if [[ -e "${evaluate}/OUTCOMES.json" ]]; then return; fi
    mkdir -p "${evaluate}"
    JG_INFERENCE_DRAW="${draw}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m joint_geometry.main validate \
        --config "${JOINT}" \
        --config "${CONFIGS}/${arm}.yaml" \
        --config "${CONFIGS}/evaluate_local.yaml" \
        --seed_everything "${seed}" \
        --model.geometry_weight "${weight}" \
        --model.sampler.init_args.geometry_dimension "${GEOMETRY_DIMENSION}" \
        --model.model.init_args.encoder.init_args.geometry_dimension "${GEOMETRY_DIMENSION}" \
        --model.model.init_args.encoder.init_args.message_graph "${MESSAGE_GRAPH}" \
        --model.store_validation_structures_path "${evaluate}/structures.xyz" \
        --data.geometry_dir "${ARTIFACT}" \
        --trainer.logger.init_args.save_dir "${evaluate}" \
        --trainer.logger.init_args.name "" \
        --trainer.logger.init_args.version "" \
        --ckpt_path "${checkpoint}" 2>&1 | tee "${evaluate}/validate.log"
    score_saved_validation "${arm}" "${seed}" "${draw}" "${evaluate}"
}

evaluate_direct() {
    local arm="$1" seed="$2" mode="$3" draw="${4:-0}"
    local run="${RUN_ROOT}/${mode}/${arm}_seed${seed}"
    local checkpoint
    checkpoint="$(latest_checkpoint "${run}")"
    local evaluate="${run}/eval/draw${draw}"
    if [[ -e "${evaluate}/OUTCOMES.json" ]]; then return; fi
    mkdir -p "${evaluate}"
    JG_INFERENCE_DRAW="${draw}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m omg.main validate \
        --config "${BASE}" \
        --config "${DIRECT}" \
        --config "${CONFIGS}/${arm}.yaml" \
        --config "${CONFIGS}/evaluate_local.yaml" \
        --seed_everything "${seed}" \
        --model.store_validation_structures_path "${evaluate}/structures.xyz" \
        --data.train_dataset.init_args.file_path "${DATA_DIR}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${DATA_DIR}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${DATA_DIR}/test.lmdb" \
        --model.model.init_args.encoder.init_args.message_graph "${MESSAGE_GRAPH}" \
        --trainer.logger.init_args.save_dir "${evaluate}" \
        --trainer.logger.init_args.name "" \
        --trainer.logger.init_args.version "" \
        --ckpt_path "${checkpoint}" 2>&1 | tee "${evaluate}/validate.log"
    score_saved_validation "${arm}" "${seed}" "${draw}" "${evaluate}"
}

score_saved_validation() {
    local arm="$1" seed="$2" draw="$3" evaluate="$4"
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
        --arm "${arm}" \
        --seed "${seed}" \
        --draw "${draw}"
}

run_smoke() {
    local representation="${JG_SMOKE_REPRESENTATION:-cn-rdf4}"
    local smoke_graph="${JG_MESSAGE_GRAPH:-periodic_distance}"
    if [[ -z "${JG_SMOKE_REPRESENTATION:-}" && -e "${REPORT_ROOT}/DESCRIPTOR-GATE.json" ]]; then
        representation="$("${PYTHON}" -c "import json; r=json.load(open('${REPORT_ROOT}/DESCRIPTOR-GATE.json')); assert r['verdict']['passed']; print(r['verdict']['promoted'])")"
    fi
    if [[ -z "${JG_MESSAGE_GRAPH:-}" && -e "${REPORT_ROOT}/BACKBONE-GATE.json" ]]; then
        smoke_graph="$("${PYTHON}" -c "import json; r=json.load(open('${REPORT_ROOT}/BACKBONE-GATE.json')); assert r['verdict']['passed']; print(r['verdict']['selected_backbone'])")"
    fi
    local smoke_artifact="${SMOKE_ARTIFACT_OVERRIDE:-/tmp/joint_geometry-smoke-cgfm64/${representation}}"
    local memorize_artifact="${MEMORIZE_ARTIFACT_OVERRIDE:-/tmp/joint_geometry-memorize10/${representation}}"
    CUDA_VISIBLE_DEVICES="" "${PYTHON}" -m joint_geometry.scripts.build_descriptor \
        --data-dir "${SMOKE_DATA_DIR}" \
        --out-dir "${smoke_artifact}" \
        --representation "${representation}" \
        --sample-atoms 256 \
        --batch-structures 8 \
        --device cpu
    CUDA_VISIBLE_DEVICES="" "${PYTHON}" -m joint_geometry.scripts.build_descriptor \
        --data-dir "${MEMORIZE_DATA_DIR}" \
        --out-dir "${memorize_artifact}" \
        --representation "${representation}" \
        --sample-atoms 256 \
        --batch-structures 10 \
        --device cpu
    local smoke_dimension
    smoke_dimension="$("${PYTHON}" -c "import json; print(json.load(open('${smoke_artifact}/manifest.json'))['dimension'])")"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m joint_geometry.main fit \
        --config "${JOINT}" \
        --config "${CONFIGS}/J.yaml" \
        --data.train_dataset.init_args.file_path "${SMOKE_DATA_DIR}/train.lmdb" \
        --data.val_dataset.init_args.file_path "${SMOKE_DATA_DIR}/val.lmdb" \
        --data.pred_dataset.init_args.file_path "${SMOKE_DATA_DIR}/test.lmdb" \
        --data.geometry_dir "${smoke_artifact}" \
        --data.batch_size 32 \
        --data.num_workers 0 \
        --data.persistent_workers false \
        --data.prefetch_factor null \
        --model.sampler.init_args.geometry_dimension "${smoke_dimension}" \
        --model.model.init_args.encoder.init_args.geometry_dimension "${smoke_dimension}" \
        --model.model.init_args.encoder.init_args.message_graph "${smoke_graph}" \
        --model.si.init_args.integration_time_steps 5 \
        --model.number_cpus 2 \
        --trainer.fast_dev_run true

    local run="${RUN_ROOT}/memorise10/${representation}_${smoke_graph}/J_seed0"
    if [[ ! -e "${run}/COMPLETE" ]]; then
        mkdir -p "${run}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m joint_geometry.main fit \
            --config "${JOINT}" \
            --config "${CONFIGS}/J.yaml" \
            --config "${CONFIGS}/memorize.yaml" \
            --data.train_dataset.init_args.file_path "${MEMORIZE_DATA_DIR}/train.lmdb" \
            --data.val_dataset.init_args.file_path "${MEMORIZE_DATA_DIR}/val.lmdb" \
            --data.pred_dataset.init_args.file_path "${MEMORIZE_DATA_DIR}/test.lmdb" \
            --data.geometry_dir "${memorize_artifact}" \
            --data.batch_size 10 \
            --data.num_workers 0 \
            --data.persistent_workers false \
            --data.prefetch_factor null \
            --model.sampler.init_args.geometry_dimension "${smoke_dimension}" \
            --model.model.init_args.encoder.init_args.geometry_dimension "${smoke_dimension}" \
            --model.model.init_args.encoder.init_args.message_graph "${smoke_graph}" \
            --trainer.accumulate_grad_batches 1 \
            --trainer.max_epochs 100 \
            --trainer.max_steps 100 \
            --trainer.logger.init_args.save_dir "${run}" \
            --trainer.logger.init_args.name "" \
            --trainer.logger.init_args.version "" 2>&1 | tee "${run}/train.log"
        "${PYTHON}" -m joint_geometry.scripts.check_memorization "${run}/metrics.csv"
        touch "${run}/COMPLETE"
    fi
}

calibrate_weight() {
    local report="${REPORT_ROOT}/GEOMETRY-WEIGHT.json"
    if [[ ! -e "${report}" ]]; then
        joint_fit J 0 20 null calibration
        "${PYTHON}" -m joint_geometry.scripts.calibration_weight \
            "$(latest_checkpoint "${RUN_ROOT}/calibration/J_seed0")" \
            --out "${report}"
    fi
    "${PYTHON}" -c "import json; print(json.load(open('${report}'))['geometry_weight'])"
}

run_oracle() {
    local weight="$1"
    direct_fit D 0 "${ORACLE_STEPS}" oracle
    joint_fit O 0 "${ORACLE_STEPS}" "${weight}" oracle
    evaluate_direct D 0 oracle 0
    evaluate_joint O 0 oracle "${weight}" 0
    "${PYTHON}" -m joint_geometry.scripts.local_report \
        --run-root "${RUN_ROOT}/oracle" \
        --out "${REPORT_ROOT}/ORACLE-GATE.json"
    "${PYTHON}" -c "import json,sys; r=json.load(open('${REPORT_ROOT}/ORACLE-GATE.json')); sys.exit(0 if r['oracle_passed'] else 1)"
}

run_single_arm() {
    local weight="$1" arm="$2"
    case "${arm}" in
        D|R)
            direct_fit "${arm}" 0 "${ORACLE_STEPS}" oracle
            evaluate_direct "${arm}" 0 oracle 0
            ;;
        *)
            joint_fit "${arm}" 0 "${ORACLE_STEPS}" "${weight}" oracle
            evaluate_joint "${arm}" 0 oracle "${weight}" 0
            ;;
    esac
}

run_screen() {
    local weight="$1"
    local seed arm
    for seed in "${SEEDS[@]}"; do
        direct_fit D "${seed}" "${LOCAL_STEPS}" screen
        evaluate_direct D "${seed}" screen 0
        for arm in H P J; do
            joint_fit "${arm}" "${seed}" "${LOCAL_STEPS}" "${weight}" screen
            evaluate_joint "${arm}" "${seed}" screen "${weight}" 0
        done
        if [[ "${RUN_R}" == 1 ]]; then
            direct_fit R "${seed}" "${LOCAL_STEPS}" screen
            evaluate_direct R "${seed}" screen 0
        fi
        "${PYTHON}" -m joint_geometry.scripts.local_report \
            --run-root "${RUN_ROOT}/screen" \
            --out "${REPORT_ROOT}/LOCAL-GATE.json"
        if [[ "${seed}" == "${SEEDS[0]}" && "${#SEEDS[@]}" -gt 1 ]]; then
            "${PYTHON}" -c "import json,sys; r=json.load(open('${REPORT_ROOT}/LOCAL-GATE.json')); c=r['contrasts']; required=('D','P','H','R'); sys.exit(0 if all(c[a] is not None and c[a]['difference_points'] > 0 for a in required) else 1)"
        fi
    done
    if [[ "${#SEEDS[@]}" -ge 3 ]]; then
        "${PYTHON}" -c "import json,sys; r=json.load(open('${REPORT_ROOT}/LOCAL-GATE.json')); sys.exit(0 if r['verdict']['promote_to_a100'] else 1)"
        local draw
        for draw in 1 2 3 4; do
            for seed in "${SEEDS[@]}"; do
                evaluate_direct D "${seed}" screen "${draw}"
                for arm in H P J; do
                    evaluate_joint "${arm}" "${seed}" screen "${weight}" "${draw}"
                done
                if [[ "${RUN_R}" == 1 ]]; then
                    evaluate_direct R "${seed}" screen "${draw}"
                fi
            done
        done
        "${PYTHON}" -m joint_geometry.scripts.local_report \
            --run-root "${RUN_ROOT}/screen" \
            --out "${REPORT_ROOT}/LOCAL-GATE.json"
        "${PYTHON}" -c "import json,sys; r=json.load(open('${REPORT_ROOT}/LOCAL-GATE.json')); sys.exit(0 if r['verdict']['promote_to_a100'] else 1)"
    fi
}

if [[ "${MODE}" == all || "${MODE}" == smoke ]]; then run_smoke; fi

if [[ "${MODE}" == all || "${MODE}" == calibrate || "${MODE}" == oracle || "${MODE}" == screen
      || "${MODE}" == arm ]]; then
    if [[ ! -e "${REPORT_ROOT}/DESCRIPTOR-GATE.json" || ! -e "${REPORT_ROOT}/BACKBONE-GATE.json" ]]; then
        echo "Descriptor and backbone gates must pass before model screens." >&2
        exit 1
    fi
    "${PYTHON}" -c "import json,sys; paths=['${REPORT_ROOT}/DESCRIPTOR-GATE.json','${REPORT_ROOT}/BACKBONE-GATE.json']; sys.exit(0 if all(json.load(open(p))['verdict']['passed'] for p in paths) else 1)"
    if [[ -z "${ARTIFACT_OVERRIDE}" ]]; then
        PROMOTED="$("${PYTHON}" -c "import json; print(json.load(open('${REPORT_ROOT}/DESCRIPTOR-GATE.json'))['verdict']['promoted'])")"
        ARTIFACT="${ARTIFACT_ROOT}/${PROMOTED}"
    fi
    if [[ -z "${DIMENSION_OVERRIDE}" ]]; then
        GEOMETRY_DIMENSION="$("${PYTHON}" -c "import json; print(json.load(open('${ARTIFACT}/manifest.json'))['dimension'])")"
    fi
    if [[ -z "${JG_MESSAGE_GRAPH:-}" ]]; then
        MESSAGE_GRAPH="$("${PYTHON}" -c "import json; print(json.load(open('${REPORT_ROOT}/BACKBONE-GATE.json'))['verdict']['selected_backbone'])")"
    fi
    calibrate_weight
    WEIGHT="$("${PYTHON}" -c "import json; print(json.load(open('${REPORT_ROOT}/GEOMETRY-WEIGHT.json'))['geometry_weight'])")"
    if [[ "${MODE}" == arm ]]; then run_single_arm "${WEIGHT}" "${SINGLE_ARM}"; fi
    if [[ "${MODE}" == all || "${MODE}" == oracle ]]; then run_oracle "${WEIGHT}"; fi
    if [[ "${MODE}" == all || "${MODE}" == screen ]]; then
        if [[ ! -e "${REPORT_ROOT}/ORACLE-GATE.json" ]]; then
            echo "The passing oracle gate is required before the real-vs-dummy screen." >&2
            exit 1
        fi
        "${PYTHON}" -c "import json,sys; r=json.load(open('${REPORT_ROOT}/ORACLE-GATE.json')); sys.exit(0 if r['oracle_passed'] else 1)"
        run_screen "${WEIGHT}"
    fi
fi

if [[ "${MODE}" == all || "${MODE}" == report ]]; then
    "${PYTHON}" -m joint_geometry.scripts.local_report \
        --run-root "${RUN_ROOT}/screen" \
        --out "${REPORT_ROOT}/LOCAL-GATE.json"
fi

#!/usr/bin/env bash
#
# Phase 2: the short local runs on one RTX PRO 6000, and nothing more than a screen.
#
#   bash direct_geometry/scripts/run_local.sh smoke      # every arm, two epochs on 100 structures. Do not skip.
#   bash direct_geometry/scripts/run_local.sh memorise   # every arm memorises 100 structures. Disqualifies non-optimisers.
#   bash direct_geometry/scripts/run_local.sh screen     # every arm, 100 epochs, one shared seed.
#   bash direct_geometry/scripts/run_local.sh paired     # A and D, 200 epochs, seeds 0 and 1.
#   bash direct_geometry/scripts/run_local.sh mse         # fixed-time denoising error of every finished screen/paired run.
#   bash direct_geometry/scripts/run_local.sh gate       # read Gate DG3 and stamp LOCAL-GATE.json.
#
# The memorisation mode was called `overfit` and wrote to runs/overfit. Its first version stopped mid-ascent and sampled at a
# quarter of the production budget, so its numbers answer a different question and are kept there as the evidence for why the
# check changed. Gate DG3 reads `memorise` and declines anything else it finds, so the old directory cannot be read as the
# new result.
#
# THE FOUR ARMS
#
# Two binary factors, crossed. The message graph and the node descriptor are separately switchable precisely so that a win
# can be attributed to one of them:
#
#   arm  message_graph       feature_mode  what it is
#   ---  ------------------  ------------  ------------------------------------------------------------------
#   A    fc                  none          the baseline, exactly: same class, zeroed additions, stock topology
#   B    periodic_distance   none          the corrected periodic multiedge graph alone
#   C    fc                  promoted      the node descriptor alone
#   D    periodic_distance   promoted      both
#   E    fc_distance         none          the edge length channel alone, on the baseline's own topology
#
# All five are the same class with the same parameter count, so a contrast is not confounded with model size. Arm A is a
# re-run rather than a quoted number: the published 26.82 per cent came from stock CSPNetFull, and a cross-arm contrast has
# to hold the code path fixed.
#
# WHY ARM E EXISTS
#
# B changes two things at once: it adds periodic multiedges *and* it hands every edge its true Cartesian length. Those have
# different mechanisms and different consequences. The baseline edge MLP is fed a sinusoidal embedding of the fractional
# offset alongside the lattice Gram matrix, so a length is a bilinear form it must learn to construct; the project's own
# classifier gained 7.29 points when simply given the distances. E supplies that product on the unchanged topology, so
# E-A is the distance channel alone and B-E is the periodic topology alone. Without E, a B-A win is unattributable, and
# the cheapest explanation -- that the trunk merely cannot multiply -- would be the one left untested.
#
# `promoted` is read out of the DG1/DG2 probe report rather than named here. The probes decide whether the descriptor is
# `radial` or `both`, and letting this script carry its own default would let the two disagree silently.
#
# WHAT THESE RUNS CAN AND CANNOT DECIDE
#
# The baseline reaches 26.82 per cent at 1200 epochs, so a 200-epoch match rate is a point on a rising curve. It can catch an
# arm that cannot optimise, whose channels interfere, whose cost breaks the ceiling, or whose topology hurts. It cannot rank
# two arms that are close. Gate DG3 is written as a blunt screen for that reason, and passing it authorises the A100 bundle
# rather than supporting any claim.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:-}"
case "${MODE}" in
    smoke|memorise|screen|paired|mse|gate) ;;
    *) echo "usage: $0 {smoke|memorise|screen|paired|mse|gate}" >&2; exit 2 ;;
esac

if [[ -n "${DG_PYTHON:-}" ]]; then PYTHON="${DG_PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then PYTHON="python"
else PYTHON="python3"
fi

source direct_geometry/scripts/wandb_logging.sh

BASE_CONFIG="${DG_BASE_CONFIG:-cgfm/configs/atomwise_mpts52.yaml}"
CONFIGS="direct_geometry/configs"
DATA_DIR="${DG_DATA_DIR:-omg/data/mpts_52}"
SMOKE_DATA_DIR="${DG_SMOKE_DATA_DIR:-cgfm/overfit_data}"
ROOT="${DG_RUN_ROOT:-direct_geometry/runs}"
REPORTS="${DG_REPORT_DIR:-direct_geometry/reports}"
PROBE_REPORT="${DG_PROBE_REPORT:-${REPORTS}/DG1-DG2-PROBES.json}"
GPU="${DG_GPU:-0}"
SCREEN_EPOCHS="${DG_SCREEN_EPOCHS:-100}"
PAIRED_EPOCHS="${DG_PAIRED_EPOCHS:-200}"
IFS=',' read -r -a PAIRED_SEEDS <<< "${DG_PAIRED_SEEDS:-0,1}"
SCREEN_SEED="${DG_SCREEN_SEED:-0}"
IFS=',' read -r -a SCREEN_ARMS <<< "${DG_SCREEN_ARMS:-A,B,C,D,E}"
IFS=',' read -r -a PAIRED_ARMS <<< "${DG_PAIRED_ARMS:-A,D}"
# Every arm, not the A, B, D subset the plan named. Arm C was assumed to be bracketed by A and D, which is an assumption a
# thirty-minute run can remove; and adding arms to a floor every arm must clear makes the gate stricter, never looser.
IFS=',' read -r -a MEMORISE_ARMS <<< "${DG_MEMORISE_ARMS:-A,B,C,D,E}"
DRY_RUN="${DG_DRY_RUN:-0}"

# The descriptor the probes promoted. Read once, here, so that every arm in every mode uses the same one and a stale probe
# report is an error rather than a silently different experiment.
promoted_mode() {
    if [[ -n "${DG_PROMOTED:-}" ]]; then
        echo "${DG_PROMOTED}"
        return 0
    fi
    "${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" --promoted
}

# Per-arm switches: the two factors, and nothing else.
arm_graph() {
    case "$1" in
        A|C) echo "fc" ;;
        B|D) echo "periodic_distance" ;;
        E)   echo "fc_distance" ;;
    esac
}

arm_features() {
    case "$1" in
        A|B|E) echo "none" ;;
        C|D)   echo "$2" ;;
    esac
}

for arm in "${SCREEN_ARMS[@]}" "${PAIRED_ARMS[@]}" "${MEMORISE_ARMS[@]}"; do
    case "${arm}" in
        A|B|C|D|E) ;;
        *) echo "unknown arm '${arm}'; choose from A, B, C, D, E" >&2; exit 2 ;;
    esac
done

if [[ "${MODE}" == "smoke" || "${MODE}" == "memorise" ]]; then
    CHECK_DATA="${SMOKE_DATA_DIR}"
else
    CHECK_DATA="${DATA_DIR}"
fi
if [[ "${MODE}" != "gate" && "${DRY_RUN}" != "1" ]]; then
    for split in train val test; do
        [[ -s "${CHECK_DATA}/${split}.lmdb" ]] || { echo "missing ${CHECK_DATA}/${split}.lmdb" >&2; exit 1; }
    done
fi

# One run: an arm at a seed. Writes the resolved command beside the run so that the exact invocation is recoverable from the
# artefact rather than from this file, which will have moved on.
run_one() {
    local arm="$1" seed="$2" promoted="$3"
    local graph features run_dir
    graph="$(arm_graph "${arm}")"
    features="$(arm_features "${arm}" "${promoted}")"
    run_dir="${ROOT}/${MODE}/${arm}_seed${seed}"

    local overlay data_dir="${DATA_DIR}" epochs=() extra=()
    case "${MODE}" in
        smoke)    overlay="${CONFIGS}/smoke.yaml";   data_dir="${SMOKE_DATA_DIR}" ;;
        memorise) overlay="${CONFIGS}/overfit.yaml"; data_dir="${SMOKE_DATA_DIR}"
                 # The training split is also the validation split: a low match rate here is an optimisation failure and
                 # cannot be anything else.
                 extra=(--data.val_dataset.init_args.file_path "${SMOKE_DATA_DIR}/train.lmdb") ;;
        screen)  overlay="${CONFIGS}/local.yaml";   epochs=(--trainer.max_epochs "${SCREEN_EPOCHS}") ;;
        paired)  overlay="${CONFIGS}/local.yaml";   epochs=(--trainer.max_epochs "${PAIRED_EPOCHS}") ;;
    esac

    # Emitted after the CSV logger's own arguments: jsonargparse applies --trainer.logger.init_args.* to the most
    # recently named logger, so the append and everything following it belong to wandb.
    wandb_arguments "${run_dir}" "${MODE}" "${arm}" "${graph}" "${features}" "${seed}" "${overlay}"

    local command=("${PYTHON}" -m omg.main fit
        --config "${BASE_CONFIG}"
        --config "${CONFIGS}/encoder.yaml"
        --config "${overlay}"
        --model.model.init_args.encoder.init_args.message_graph "${graph}"
        --model.model.init_args.encoder.init_args.feature_mode "${features}"
        --seed_everything "${seed}"
        --data.train_dataset.init_args.file_path "${data_dir}/train.lmdb"
        --data.val_dataset.init_args.file_path "${data_dir}/val.lmdb"
        --data.pred_dataset.init_args.file_path "${data_dir}/test.lmdb"
        "${extra[@]+"${extra[@]}"}"
        "${epochs[@]+"${epochs[@]}"}"
        --trainer.logger.init_args.save_dir "${run_dir}"
        --trainer.logger.init_args.name ""
        --trainer.logger.init_args.version ""
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}")

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=${GPU} ${command[*]}"
        return 0
    fi

    # A finished run is left alone. A mode is often re-entered to add an arm, and silently retraining the ones already done
    # would throw away hours and, worse, replace a number that has already been read.
    if [[ -s "${run_dir}/metrics.csv" && -s "${run_dir}/COMPLETE" ]]; then
        echo "$(date -u +%H:%M) ${MODE} arm ${arm} seed ${seed} already complete; keeping ${run_dir}"
        return 0
    fi

    rm -rf "${run_dir}"
    mkdir -p "${run_dir}"
    printf '%s\n' "${command[@]}" > "${run_dir}/COMMAND"
    echo "$(date -u +%H:%M) ${MODE} arm ${arm} seed ${seed}: graph ${graph}, features ${features}"

    local status=0
    CUDA_VISIBLE_DEVICES="${GPU}" WANDB_MODE="${WANDB_RESOLVED_MODE}" \
        "${command[@]}" > "${run_dir}.log" 2>&1 || status=$?
    if (( status == 0 )); then
        date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/COMPLETE"
    fi
    echo "$(date -u +%H:%M) ${MODE} arm ${arm} seed ${seed} exited ${status}; see ${run_dir}.log"
    return ${status}
}

if [[ "${MODE}" == "gate" ]]; then
    exec "${PYTHON}" -m direct_geometry.scripts.local_gate \
        --run-root "${ROOT}" --probe-report "${PROBE_REPORT}" --out "${REPORTS}/LOCAL-GATE.json"
fi

# Gate DG3 is stated on the denoising error as well as the match rate, and that error is measured after training from the
# saved checkpoint rather than during it. Only the screen and paired runs are measured: smoke and memorise exist to catch
# breakage and non-optimisers, and a denoising error on a hundred memorised structures means nothing.
if [[ "${MODE}" == "mse" ]]; then
    measured=0
    mse_failures=()
    for run_dir in "${ROOT}"/screen/*/ "${ROOT}"/paired/*/; do
        [[ -d "${run_dir}" ]] || continue
        run_dir="${run_dir%/}"
        [[ -s "${run_dir}/COMPLETE" ]] || { echo "skipping $(basename "${run_dir}"): not complete"; continue; }
        if [[ -s "${run_dir}/TARGET-MSE.json" ]]; then
            echo "$(date -u +%H:%M) $(basename "${run_dir}") already measured"
            continue
        fi
        echo "$(date -u +%H:%M) measuring $(basename "${run_dir}")"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m direct_geometry.scripts.target_mse \
            --run-dir "${run_dir}" --data-dir "${DATA_DIR}" >> "${run_dir}.mse.log" 2>&1 \
            || mse_failures+=("$(basename "${run_dir}")")
        measured=$((measured + 1))
    done
    if (( ${#mse_failures[@]} > 0 )); then
        echo "$(date -u +%H:%M) target error failed for: ${mse_failures[*]}" >&2
        exit 1
    fi
    echo "$(date -u +%H:%M) mse: measured ${measured} run(s)."
    exit 0
fi

PROMOTED="$(promoted_mode)"
echo "$(date -u +%H:%M) ${MODE}: promoted descriptor '${PROMOTED}', GPU ${GPU}"
wandb_resolve_mode
wandb_announce

case "${MODE}" in
    smoke)   RUNS=(); for arm in "${SCREEN_ARMS[@]}"; do RUNS+=("${arm}:0"); done ;;
    memorise) RUNS=(); for arm in "${MEMORISE_ARMS[@]}"; do RUNS+=("${arm}:0"); done ;;
    screen)  RUNS=(); for arm in "${SCREEN_ARMS[@]}"; do RUNS+=("${arm}:${SCREEN_SEED}"); done ;;
    paired)  RUNS=()
             # Seed-major: a launch cut short then holds whole seeds, and a paired difference needs both arms of a seed.
             for seed in "${PAIRED_SEEDS[@]}"; do
                 for arm in "${PAIRED_ARMS[@]}"; do RUNS+=("${arm}:${seed}"); done
             done ;;
esac

# One card, so strictly sequential. Failures are counted and re-raised at the end rather than aborting: the remaining runs
# are still worth having, and a mode that exits zero with a missing run would be read as a complete result.
failures=()
for run in "${RUNS[@]}"; do
    IFS=':' read -r arm seed <<< "${run}"
    run_one "${arm}" "${seed}" "${PROMOTED}" || failures+=("${arm} seed ${seed}")
done

if (( ${#failures[@]} > 0 )); then
    echo "$(date -u +%H:%M) ${#failures[@]} of ${#RUNS[@]} runs failed: ${failures[*]}" >&2
    exit 1
fi
echo "$(date -u +%H:%M) ${MODE}: all ${#RUNS[@]} runs finished."

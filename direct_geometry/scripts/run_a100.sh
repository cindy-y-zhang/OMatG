#!/usr/bin/env bash
#
# Phase 3: the full sweep on eight A100s.
#
#   bash direct_geometry/scripts/run_a100.sh verify    # environment, tests, dry run. Do not skip.
#   bash direct_geometry/scripts/run_a100.sh smoke     # every arm, two epochs, sequentially on one card.
#   bash direct_geometry/scripts/run_a100.sh launch    # the sweep: every arm at every seed, one process per GPU.
#   bash direct_geometry/scripts/run_a100.sh evaluate  # repeated validation draws and the locked test read.
#   bash direct_geometry/scripts/run_a100.sh report    # paired-seed contrasts and the deployment verdict.
#
# THE SWEEP
#
# Five arms crossed with three shared seeds, fifteen independent runs of 1,600 epochs. The arms are the local screen's:
#
#   arm  message_graph       feature_mode  isolates
#   ---  ------------------  ------------  -----------------------------------------------------
#   A    fc                  none          nothing: the baseline code path, additions zeroed
#   B    periodic_distance   none          periodic multiedges and true edge lengths, together
#   C    fc                  promoted      the node descriptor
#   D    periodic_distance   promoted      both
#   E    fc_distance         none          the edge length channel alone, baseline topology
#
# Arm E costs no wall clock. Fifteen runs over eight GPUs is two waves, and so was the twelve-run version, so E rides
# along in the second wave's spare capacity. It is what makes B-A attributable: B changes the topology and the edge length
# channel at once, and the cheap explanation -- that the trunk simply cannot form a distance from a sinusoidal embedding of
# a fractional offset times a lattice Gram matrix -- is the one E tests.
#
# NO DDP
#
# One process per GPU, each training its own run to completion. Fifteen single-GPU runs is the same arithmetic as fifteen
# eight-GPU runs and needs no gradient synchronisation, no per-rank seeding care, and no reasoning about whether the
# effective batch changed. DDP would also silently alter the effective batch, which is the one thing every arm must share.
#
# WHAT THIS CAN DECIDE, AND WHAT IT CANNOT
#
# This is the evidence. The local gate was a screen. But it is still fifteen runs, and the contrast that matters is read
# paired within seed with a standard error attached, because seed-to-seed spread on MPTS-52 match rate is of the same order
# as the effect being looked for. A single-seed difference from this sweep is not a result either.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:-}"
case "${MODE}" in
    verify|smoke|launch|evaluate|report) ;;
    *) echo "usage: $0 {verify|smoke|launch|evaluate|report}" >&2; exit 2 ;;
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
ROOT="${DG_RUN_ROOT:-direct_geometry/a100_runs}"
REPORTS="${DG_REPORT_DIR:-direct_geometry/reports}"
PROBE_REPORT="${DG_PROBE_REPORT:-${REPORTS}/DG1-DG2-PROBES.json}"
GATE_STAMP="${DG_GATE_STAMP:-${REPORTS}/LOCAL-GATE.json}"
EPOCHS="${DG_EPOCHS:-1600}"
# Which mode's runs the sampling jobs read. Named rather than hard-coded so that a re-run of the smoke sweep can also be
# sampled end to end, which is the cheapest way to prove the evaluation path works before the real one finishes.
SWEEP_MODE="${DG_SWEEP_MODE:-launch}"
IFS=',' read -r -a ARMS <<< "${DG_ARMS:-A,B,C,D,E}"
IFS=',' read -r -a SEEDS <<< "${DG_SEEDS:-0,1,2}"
IFS=',' read -r -a GPUS <<< "${DG_GPUS:-0,1,2,3,4,5,6,7}"
DRY_RUN="${DG_DRY_RUN:-0}"

for arm in "${ARMS[@]}"; do
    case "${arm}" in
        A|B|C|D|E) ;;
        *) echo "unknown arm '${arm}'; choose from A, B, C, D, E" >&2; exit 2 ;;
    esac
done

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

# The descriptor the DG1/DG2 probes promoted, read from their stamp through the shared reader rather than parsed here, so
# that this launcher, the local one and the packager cannot disagree about which experiment is being run.
promoted_mode() {
    if [[ -n "${DG_PROMOTED:-}" ]]; then echo "${DG_PROMOTED}"; return 0; fi
    "${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" --promoted
}

if [[ "${MODE}" == "verify" ]]; then
    echo "== environment =="
    "${PYTHON}" -c "
import torch
print(f'torch {torch.__version__}, cuda {torch.version.cuda}, {torch.cuda.device_count()} device(s)')
for index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    print(f'  gpu {index}: {name}, {total:.0f} GiB')
"
    echo
    echo "== data =="
    for split in train val test; do
        [[ -s "${DATA_DIR}/${split}.lmdb" ]] || { echo "missing ${DATA_DIR}/${split}.lmdb" >&2; exit 1; }
        echo "  ${DATA_DIR}/${split}.lmdb $(du -h "${DATA_DIR}/${split}.lmdb" | cut -f1)"
    done
    for split in train val test; do
        [[ -s "${SMOKE_DATA_DIR}/${split}.lmdb" ]] || { echo "missing ${SMOKE_DATA_DIR}/${split}.lmdb" >&2; exit 1; }
    done
    echo "  smoke split present"
    echo
    echo "== the gates this sweep rests on =="
    # Informational: nothing is required here, so a missing DG3 is reported rather than fatal. `launch` requires it.
    "${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" || true
    echo
    echo "== tests =="
    "${PYTHON}" -m pytest direct_geometry/tests -q
    echo
    echo "== dry run =="
    DG_DRY_RUN=1 "${BASH_SOURCE[0]}" launch 2>/dev/null || DG_DRY_RUN=1 bash "${BASH_SOURCE[0]}" launch
    exit 0
fi

if [[ "${MODE}" == "report" ]]; then
    exec "${PYTHON}" -m direct_geometry.scripts.a100_report \
        --run-root "${ROOT}" --probe-report "${PROBE_REPORT}" --out "${REPORTS}/A100-REPORT.json"
fi

PROMOTED="$(promoted_mode)"

# Refused rather than warned. The stamps are the whole reason to believe this sweep is worth fifteen GPU-weeks, and a sweep
# launched without them is one whose result cannot be attributed to a predeclared design.
if [[ "${MODE}" == "launch" && "${DRY_RUN}" != "1" && "${DG_IGNORE_GATE:-0}" != "1" ]]; then
    "${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" --require dg0,probes,dg3 || {
        echo
        echo "Set DG_IGNORE_GATE=1 only to reproduce a superseded launch deliberately." >&2
        exit 1
    }
fi

# One run. Bound to a single GPU by the caller, so this never chooses a device itself.
launch_one() {
    local arm="$1" seed="$2" gpu="$3"
    local graph features run_dir overlay data_dir epochs
    graph="$(arm_graph "${arm}")"
    features="$(arm_features "${arm}" "${PROMOTED}")"
    run_dir="${ROOT}/${MODE}/${arm}_seed${seed}"

    if [[ "${MODE}" == "smoke" ]]; then
        overlay="${CONFIGS}/smoke.yaml"; data_dir="${SMOKE_DATA_DIR}"; epochs=2
    else
        overlay="${CONFIGS}/a100.yaml"; data_dir="${DATA_DIR}"; epochs="${EPOCHS}"
    fi

    wandb_arguments "${run_dir}" "${MODE}" "${arm}" "${graph}" "${features}" "${seed}" "$(basename "${overlay}")"

    local command=("${PYTHON}" -m omg.main fit
        --config "${BASE_CONFIG}"
        --config "${CONFIGS}/encoder.yaml"
        --config "${overlay}"
        --model.model.init_args.encoder.init_args.message_graph "${graph}"
        --model.model.init_args.encoder.init_args.feature_mode "${features}"
        --seed_everything "${seed}"
        --trainer.max_epochs "${epochs}"
        --data.train_dataset.init_args.file_path "${data_dir}/train.lmdb"
        --data.val_dataset.init_args.file_path "${data_dir}/val.lmdb"
        --data.pred_dataset.init_args.file_path "${data_dir}/test.lmdb"
        --trainer.logger.init_args.save_dir "${run_dir}"
        --trainer.logger.init_args.name ""
        --trainer.logger.init_args.version ""
        "${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}")

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "gpu ${gpu}: ${arm} seed ${seed} (${graph}, ${features})"
        printf '  %s\n' "${command[*]}"
        return 0
    fi

    # A finished run is never retrained. These are days each, and silently replacing a number that has already been read
    # into a report is worse than the wasted time.
    if [[ -s "${run_dir}/metrics.csv" && -s "${run_dir}/COMPLETE" ]]; then
        echo "$(date -u +%H:%M) ${arm} seed ${seed} already complete; keeping ${run_dir}"
        return 0
    fi

    rm -rf "${run_dir}"
    mkdir -p "${run_dir}"
    printf '%s\n' "${command[@]}" > "${run_dir}/COMMAND"
    echo "$(date -u +%H:%M) gpu ${gpu} <- ${arm} seed ${seed} (${graph}, ${features})"

    local status=0
    CUDA_VISIBLE_DEVICES="${gpu}" WANDB_MODE="${WANDB_RESOLVED_MODE}" \
        "${command[@]}" > "${run_dir}.log" 2>&1 || status=$?
    if (( status == 0 )); then
        date -u +%Y-%m-%dT%H:%M:%SZ > "${run_dir}/COMPLETE"
    else
        echo "$(date -u +%H:%M) gpu ${gpu}: ${arm} seed ${seed} FAILED with ${status}; see ${run_dir}.log" >&2
    fi
    return ${status}
}

# Sampling the finished checkpoints, which is a separate GPU job from training them and is scheduled the same way. It reads
# the arm out of the run's own COMMAND file, so this never has to re-derive which arm a directory holds.
evaluate_one() {
    local arm="$1" seed="$2" gpu="$3"
    local run_dir="${ROOT}/${SWEEP_MODE}/${arm}_seed${seed}"

    if [[ ! -s "${run_dir}/COMPLETE" ]]; then
        echo "$(date -u +%H:%M) ${arm} seed ${seed} is not finished; nothing to sample" >&2
        return 1
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "gpu ${gpu}: evaluate ${arm} seed ${seed} (${run_dir})"
        return 0
    fi

    echo "$(date -u +%H:%M) gpu ${gpu} <- evaluate ${arm} seed ${seed}"
    local status=0
    DG_GPU="${gpu}" DG_PYTHON="${PYTHON}" DG_DATA_DIR="${DATA_DIR}" DG_BASE_CONFIG="${BASE_CONFIG}" \
        bash direct_geometry/scripts/evaluate.sh "${run_dir}" > "${run_dir}.eval.log" 2>&1 || status=$?
    if (( status != 0 )); then
        echo "$(date -u +%H:%M) gpu ${gpu}: evaluate ${arm} seed ${seed} FAILED with ${status}; see " \
             "${run_dir}.eval.log" >&2
    fi
    return ${status}
}

# One entry point for the scheduler, so the queue below does not have to know which kind of job it is handing out.
dispatch() {
    if [[ "${MODE}" == "evaluate" ]]; then
        evaluate_one "$@"
    else
        launch_one "$@"
    fi
}

QUEUE=()
if [[ "${MODE}" == "smoke" ]]; then
    for arm in "${ARMS[@]}"; do QUEUE+=("${arm}:0"); done
else
    # Seed-major, so that an interrupted sweep holds whole seeds and every paired contrast has both of its sides.
    for seed in "${SEEDS[@]}"; do
        for arm in "${ARMS[@]}"; do QUEUE+=("${arm}:${seed}"); done
    done
fi

# Resolved even for a dry run, so the printed command is the one that would actually be issued. Announced only for a real
# one, since the dry run's job is to show commands rather than to narrate the logging setup.
wandb_resolve_mode
[[ "${DRY_RUN}" == "1" ]] || wandb_announce
echo "$(date -u +%H:%M) ${MODE}: ${#QUEUE[@]} run(s), promoted descriptor '${PROMOTED}'"

if [[ "${MODE}" == "smoke" ]]; then
    # Sequential on one card. The point is to prove every arm runs end to end, and doing that on eight cards at once would
    # also be testing whether eight of these fit alongside each other, which is a different question.
    failures=()
    for item in "${QUEUE[@]}"; do
        IFS=':' read -r arm seed <<< "${item}"
        launch_one "${arm}" "${seed}" "${GPUS[0]}" || failures+=("${arm} seed ${seed}")
    done
    if (( ${#failures[@]} > 0 )); then
        echo "$(date -u +%H:%M) ${#failures[@]} of ${#QUEUE[@]} smoke runs failed: ${failures[*]}" >&2
        exit 1
    fi
    echo "$(date -u +%H:%M) smoke: all ${#QUEUE[@]} runs finished. The sweep is safe to launch."
    exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    index=0
    for item in "${QUEUE[@]}"; do
        IFS=':' read -r arm seed <<< "${item}"
        dispatch "${arm}" "${seed}" "${GPUS[index % ${#GPUS[@]}]}" || true
        index=$((index + 1))
    done
    echo "dry run: ${#QUEUE[@]} run(s) over ${#GPUS[@]} gpu(s)"
    exit 0
fi

# A work queue rather than fixed waves. Fifteen runs over eight cards leaves seven idle for the whole of the second wave
# if the split is fixed, and the arms do not all cost the same: the periodic ones build a neighbour list. Each card takes
# the next job when it frees, so the tail is one run long instead of one wave long.
[[ "${MODE}" == "evaluate" ]] || mkdir -p "${ROOT}/${MODE}"
STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "${STATUS_DIR}"' EXIT
printf '%s\n' "${QUEUE[@]}" > "${STATUS_DIR}/queue"

worker() {
    local gpu="$1" line
    # `flock` serialises the take so two cards cannot claim the same run. Without it the sweep would occasionally train one
    # arm twice and leave another missing, which the report would then read as an incomplete design.
    while true; do
        line="$(flock "${STATUS_DIR}/queue.lock" "${PYTHON}" - "${STATUS_DIR}/queue" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = [line for line in path.read_text().splitlines() if line.strip()]
if not lines:
    print("")
else:
    path.write_text("\n".join(lines[1:]) + ("\n" if lines[1:] else ""))
    print(lines[0])
PY
)"
        [[ -n "${line}" ]] || break
        IFS=':' read -r arm seed <<< "${line}"
        dispatch "${arm}" "${seed}" "${gpu}" || echo "${arm} seed ${seed}" >> "${STATUS_DIR}/failures"
    done
}

touch "${STATUS_DIR}/queue.lock"
for gpu in "${GPUS[@]}"; do
    worker "${gpu}" &
done
wait

if [[ -s "${STATUS_DIR}/failures" ]]; then
    count="$(wc -l < "${STATUS_DIR}/failures")"
    echo "$(date -u +%H:%M) ${count} of ${#QUEUE[@]} ${MODE} jobs failed:" >&2
    while read -r line; do echo "  ${line}" >&2; done < "${STATUS_DIR}/failures"
    echo "the finished ones are kept; re-run '$0 ${MODE}' to retry only the failures." >&2
    exit 1
fi
if [[ "${MODE}" == "evaluate" ]]; then
    echo "$(date -u +%H:%M) evaluate: all ${#QUEUE[@]} run(s) sampled. Read it with '$0 report'."
else
    echo "$(date -u +%H:%M) launch: all ${#QUEUE[@]} runs finished. Sample them with '$0 evaluate'."
fi

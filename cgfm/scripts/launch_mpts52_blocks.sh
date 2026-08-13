#!/usr/bin/env bash
# Launch the nine MPTS-52 jobs: three arms times three seeds.
#
#   cgfm/scripts/launch_mpts52_blocks.sh plan
#   cgfm/scripts/launch_mpts52_blocks.sh task <0..8>
#   cgfm/scripts/launch_mpts52_blocks.sh local
#   cgfm/scripts/launch_mpts52_blocks.sh slurm
#
# Do not launch these until Phase-0 G1/G2 have passed and the 100-structure overfit has cleared 80 per cent
# training-set match. Run from the repository root.

set -euo pipefail

PYTHON="${CGFM_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON="python"
    elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python3"
    fi
fi
ARMS=(atomwise oracle_coord joint)
SEEDS=(0 1 2)
NUM_TASKS=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
GATE_STAMP="${CGFM_GATE_STAMP:-cgfm/blocks/mpts_52/phase0_passed.json}"
OVERFIT_GATE_STAMP="${CGFM_OVERFIT_GATE_STAMP:-cgfm/blocks/mpts_52/phase1_passed.json}"

arm_of_task() { echo "${ARMS[$(( $1 / ${#SEEDS[@]} ))]}"; }
seed_of_task() { echo "${SEEDS[$(( $1 % ${#SEEDS[@]} ))]}"; }

require_blocks() {
    local missing=0
    for split in train val test; do
        if [[ ! -f "cgfm/blocks/mpts_52/${split}.npz" ]]; then
            echo "missing cgfm/blocks/mpts_52/${split}.npz" >&2
            missing=1
        fi
    done
    if [[ ! -f "cgfm/blocks/mpts_52/templates.pkl" ]]; then
        echo "missing cgfm/blocks/mpts_52/templates.pkl" >&2
        missing=1
    fi
    if [[ "${missing}" -eq 1 ]]; then
        echo "Run: cgfm/scripts/prepare_blocks_mpts52.sh" >&2
        exit 1
    fi
}

require_gates() {
    "${PYTHON}" - "${GATE_STAMP}" "${OVERFIT_GATE_STAMP}" cgfm/configs/block_mpts52.yaml <<'PY'
import json
import hashlib
import re
import sys
from pathlib import Path

phase0, phase1, config_path = map(Path, sys.argv[1:])
stamps = {}
for label, path in (("Phase-0", phase0), ("Phase-1 overfit", phase1)):
    if not path.is_file():
        raise SystemExit(f"missing {label} pass stamp {path}")
    try:
        stamp = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label} pass stamp {path}: {error}") from error
    if stamp.get("passed") is not True:
        raise SystemExit(f"{label} stamp {path} does not record a pass")
    stamps[label] = stamp

for label, path_key, digest_key in (
        ("Phase-0", "coarse", "coarse_sha256"),
        ("Phase-0", "fine", "fine_sha256"),
        ("Phase-1 overfit", "metrics", "metrics_sha256")):
    document = stamps[label]
    source = Path(document.get(path_key, ""))
    expected = document.get(digest_key)
    if not source.is_file() or not expected:
        raise SystemExit(f"{label} stamp does not identify its {path_key} source")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{label} {path_key} source changed after the gate passed: {source}")

overfit = stamps["Phase-1 overfit"]
phase0_gate = stamps["Phase-0"]
if float(phase0_gate.get("g1_threshold", 0.0)) < 0.90:
    raise SystemExit("Phase-0 stamp used a G1 threshold below 90 per cent")
if float(phase0_gate.get("g2_tolerance_points", float("inf"))) > 3.0:
    raise SystemExit("Phase-0 stamp used a G2 tolerance above three percentage points")
if float(overfit.get("threshold", 0.0)) < 0.80:
    raise SystemExit("Phase-1 stamp used an overfit threshold below 80 per cent")
match = re.search(r"(?m)^  consensus_weight:\s*([-+0-9.eE]+)\s*$", config_path.read_text())
if match is None:
    raise SystemExit(f"cannot read model.consensus_weight from {config_path}")
configured_weight = float(match.group(1))
selected_weight = float(overfit["consensus_weight"])
if configured_weight != selected_weight:
    raise SystemExit(
        f"Phase-1 selected consensus_weight={selected_weight}, but {config_path} configures {configured_weight}")
PY
}

case "${1:-plan}" in
    plan)
        for task in $(seq 0 $(( NUM_TASKS - 1 ))); do
            printf 'task %d  arm %-12s seed %s\n' "${task}" "$(arm_of_task "${task}")" "$(seed_of_task "${task}")"
        done
        ;;

    task)
        TASK="${2:?usage: $0 task <index>}"
        shift 2
        require_gates
        ARM="$(arm_of_task "${TASK}")"
        if [[ "${ARM}" != "atomwise" ]]; then
            require_blocks
        fi
        cgfm/scripts/run_mpts52_block_job.sh "${ARM}" "$(seed_of_task "${TASK}")" "$@"
        ;;

    local)
        require_gates
        require_blocks
        shift
        NUM_GPUS="${CGFM_NUM_GPUS:-$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')}"
        if [[ "${NUM_GPUS}" -lt 1 ]]; then
            echo "no CUDA devices visible" >&2
            exit 1
        fi
        echo "Running ${NUM_TASKS} jobs across ${NUM_GPUS} GPUs."
        for task in $(seq 0 $(( NUM_TASKS - 1 ))); do
            gpu=$(( task % NUM_GPUS ))
            arm="$(arm_of_task "${task}")"
            seed="$(seed_of_task "${task}")"
            log="runs-blocks/mpts52/${arm}/seed${seed}"
            mkdir -p "${log}"
            (
                flock 9
                CUDA_VISIBLE_DEVICES="${gpu}" cgfm/scripts/run_mpts52_block_job.sh "${arm}" "${seed}" "$@"
            ) 9> "/tmp/cgfm_block_gpu_${gpu}.lock" &
        done
        wait
        ;;

    slurm)
        require_gates
        require_blocks
        shift
        read -r -a SBATCH_ARGS <<< "${CGFM_SBATCH_ARGS:---time=30:00:00}"
        sbatch --array=0-$(( NUM_TASKS - 1 )) "${SBATCH_ARGS[@]}" \
               --wrap "cgfm/scripts/launch_mpts52_blocks.sh task \${SLURM_ARRAY_TASK_ID} $*"
        ;;

    *)
        echo "usage: $0 {plan|task <index>|local|slurm} [extra CLI arguments...]" >&2
        exit 2
        ;;
esac

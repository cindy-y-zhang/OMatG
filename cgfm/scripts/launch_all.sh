#!/usr/bin/env bash
# Launch the full experiment: four arms at three seeds each.
#
#   cgfm/scripts/launch_all.sh plan            # print the twelve jobs and exit
#   cgfm/scripts/launch_all.sh task <index>    # run one job, indexed 0 to 11
#   cgfm/scripts/launch_all.sh local           # run all jobs, one per visible GPU, round robin
#   cgfm/scripts/launch_all.sh slurm           # submit an sbatch array over all twelve jobs
#
# The task index is the unit of work everywhere, so a scheduler only has to map an array index onto "task <index>".
#
# Run from the repository root, after cgfm/scripts/precompute_groups.py has produced the group files for the training
# and validation splits.

set -euo pipefail

ARMS=(atomwise kmedoids shells learned)
SEEDS=(0 1 2)
NUM_TASKS=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))

arm_of_task() { echo "${ARMS[$(( $1 / ${#SEEDS[@]} ))]}"; }
seed_of_task() { echo "${SEEDS[$(( $1 % ${#SEEDS[@]} ))]}"; }

require_groups() {
    local missing=0
    for split in train val; do
        for method in kmedoids shells; do
            if [[ ! -f "cgfm/groups/mpts_52/${split}.${method}.npz" ]]; then
                echo "missing cgfm/groups/mpts_52/${split}.${method}.npz" >&2
                missing=1
            fi
        done
    done
    if [[ ${missing} -eq 1 ]]; then
        echo "Run: python -m cgfm.scripts.precompute_groups --data omg/data/mpts_52/<split>.lmdb \\" >&2
        echo "         --out-dir cgfm/groups/mpts_52   (for split in train val)" >&2
        exit 1
    fi
}

case "${1:-plan}" in
    plan)
        for task in $(seq 0 $(( NUM_TASKS - 1 ))); do
            printf 'task %2d  arm %-9s seed %s\n' "${task}" "$(arm_of_task "${task}")" "$(seed_of_task "${task}")"
        done
        ;;

    task)
        require_groups
        TASK="${2:?usage: $0 task <index>}"
        shift 2
        cgfm/scripts/run_arm.sh "$(arm_of_task "${TASK}")" "$(seed_of_task "${TASK}")" "$@"
        ;;

    local)
        require_groups
        shift
        NUM_GPUS="${CGFM_NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
        if [[ "${NUM_GPUS}" -lt 1 ]]; then
            echo "no CUDA devices visible" >&2
            exit 1
        fi
        echo "Running ${NUM_TASKS} jobs across ${NUM_GPUS} GPUs."
        for task in $(seq 0 $(( NUM_TASKS - 1 ))); do
            gpu=$(( task % NUM_GPUS ))
            arm="$(arm_of_task "${task}")"
            seed="$(seed_of_task "${task}")"
            log="runs/${arm}/seed${seed}"
            mkdir -p "${log}"
            # Serialise the jobs that share a GPU by chaining them through a per-GPU lock file.
            (
                flock 9
                CUDA_VISIBLE_DEVICES="${gpu}" cgfm/scripts/run_arm.sh "${arm}" "${seed}" "$@"
            ) 9> "/tmp/cgfm_gpu_${gpu}.lock" &
        done
        wait
        ;;

    slurm)
        require_groups
        shift
        # Site-specific sbatch options, for example partition and time limit, come from the environment.
        read -r -a SBATCH_ARGS <<< "${CGFM_SBATCH_ARGS:-}"
        sbatch --array=0-$(( NUM_TASKS - 1 )) "${SBATCH_ARGS[@]}" \
               --wrap "cgfm/scripts/launch_all.sh task \${SLURM_ARRAY_TASK_ID} $*"
        ;;

    *)
        echo "usage: $0 {plan|task <index>|local|slurm} [extra CLI arguments...]" >&2
        exit 2
        ;;
esac

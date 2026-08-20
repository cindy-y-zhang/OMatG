#!/usr/bin/env bash
#
# Run the remaining local stages back to back, one after another on the single card.
#
#   bash direct_geometry/scripts/run_local_chain.sh                    # memorise, paired, mse, gate
#   bash direct_geometry/scripts/run_local_chain.sh screen paired      # a chosen subset, in the order given
#   DG_WAIT_FOR=577447 bash direct_geometry/scripts/run_local_chain.sh # start once that process has exited
#
# This is a sequencer and nothing else. Every stage is a mode of run_local.sh, which owns its own run directories and its own
# skip-if-complete rule; this file adds no behaviour of its own beyond ordering. One GPU means the stages cannot overlap, and
# a screen run holding 80 GB of an RTX PRO 6000 means the next stage must wait rather than share.
#
# WHY A FAILURE DOES NOT STOP THE CHAIN
#
# The training stages are independent of each other: the memorisation check, the screen and the paired seeds share no inputs.
# Stopping at the first failure would mean an arm failing to memorise cancels ten hours of paired runs that did not depend on
# it, which is the wrong trade overnight. So each stage runs, failures are collected, and the exit code carries them.
#
# The reading stages are not treated specially either, and deliberately: the denoising errors skip runs that did not finish,
# and Gate DG3 already refuses on missing or interrupted runs and writes a stamp saying which. Running them after a partial
# set therefore produces an accurate account of what happened, which is more useful than no account at all.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

STAGES=("$@")
if (( ${#STAGES[@]} == 0 )); then
    STAGES=(memorise paired mse gate)
fi

for stage in "${STAGES[@]}"; do
    case "${stage}" in
        smoke|memorise|screen|paired|mse|gate) ;;
        *) echo "unknown stage '${stage}'; choose from smoke, memorise, screen, paired, mse, gate" >&2; exit 2 ;;
    esac
done

# Waiting on a process id rather than on a log line, because a log line can be written by a run that then fails, and the
# thing that must be true before the next stage starts is that the card is free.
if [[ -n "${DG_WAIT_FOR:-}" ]]; then
    echo "$(date -u +%H:%M) waiting for process ${DG_WAIT_FOR} to release the gpu"
    while kill -0 "${DG_WAIT_FOR}" 2>/dev/null; do
        sleep 60
    done
    echo "$(date -u +%H:%M) process ${DG_WAIT_FOR} has exited"
fi

echo "$(date -u +%H:%M) chain: ${STAGES[*]}"
failed=()
for stage in "${STAGES[@]}"; do
    echo
    echo "$(date -u +%H:%M) ===== ${stage} ====="
    if bash direct_geometry/scripts/run_local.sh "${stage}"; then
        echo "$(date -u +%H:%M) ${stage} finished"
    else
        status=$?
        failed+=("${stage} (exit ${status})")
        echo "$(date -u +%H:%M) ${stage} exited ${status}; continuing, since the later stages report what is missing" >&2
    fi
done

echo
if (( ${#failed[@]} > 0 )); then
    echo "$(date -u +%H:%M) chain finished with failures: ${failed[*]}" >&2
    exit 1
fi
echo "$(date -u +%H:%M) chain finished cleanly: ${STAGES[*]}"

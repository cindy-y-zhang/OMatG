#!/usr/bin/env bash
# Evaluate one trained arm at several numbers of Euler steps, for the results-versus-cost curve.
#
#   cgfm/scripts/nfe_curve.sh <arm> <seed> [extra CLI arguments...]
#
# The claim under test is that a coarse-to-fine path is easier to integrate, so the interesting comparison is not only
# the metric at the reference step count but how fast each arm reaches it. Set CGFM_NFE_VALUES to change the grid.
#
# One draw per step count by default, since this curve is about the shape rather than about best-of-n. Results land in
# <run dir>/eval/nfe<steps>/score.json, which cgfm/scripts/collect_results.py reads.
#
# Run from the repository root.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <atomwise|kmedoids|shells|learned> <seed> [extra CLI arguments...]" >&2
    exit 2
fi

ARM="$1"
SEED="$2"
shift 2

NFE_VALUES="${CGFM_NFE_VALUES:-10 25 50 100 210}"

for NFE in ${NFE_VALUES}; do
    echo "=== ${ARM} seed ${SEED} at ${NFE} Euler steps ==="
    CGFM_NFE="${NFE}" CGFM_NUM_DRAWS="${CGFM_NFE_DRAWS:-1}" cgfm/scripts/evaluate.sh "${ARM}" "${SEED}" auto "$@"
done

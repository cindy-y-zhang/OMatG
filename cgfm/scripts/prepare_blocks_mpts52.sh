#!/usr/bin/env bash
# One-time MPTS-52 block preprocessing, manifests, and Phase-0 gate commands.
#
#   cgfm/scripts/prepare_blocks_mpts52.sh
#
# Fits train-only templates, writes block tables for every split, and prints the G1/G2 ceiling commands. It does not
# run those ceilings: they are the user's to submit, and training must not start if either gate fails.
#
# Run from the repository root.

set -euo pipefail

PYTHON="${CGFM_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON="python"
    elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
    else PYTHON="python3"
    fi
fi
DATA_DIR="${CGFM_DATA_DIR:-omg/data/mpts_52}"
BLOCK_DIR="${CGFM_BLOCK_DIR:-cgfm/blocks/mpts_52}"
REPORT_DIR="${CGFM_REPORT_DIR:-cgfm/reports}"
WORKERS="${CGFM_WORKERS:-8}"

echo "== precomputing train-only block tables =="
"${PYTHON}" -m cgfm.scripts.precompute_blocks \
    --data-dir "${DATA_DIR}" \
    --out-dir "${BLOCK_DIR}" \
    --workers "${WORKERS}" \
    --type-key centre-cn

echo
echo "== Phase-0 gates (submit these; do not launch GPU training if either fails) =="
echo "  ${PYTHON} -m cgfm.scripts.readout_ceiling \\"
echo "      --data ${DATA_DIR}/val.lmdb --template-data ${DATA_DIR}/train.lmdb \\"
echo "      --type-key centre-cn --cache-dir ${BLOCK_DIR}/cache --workers ${WORKERS} \\"
echo "      --json-out ${REPORT_DIR}/readout_g1_coarse.json"
echo "  ${PYTHON} -m cgfm.scripts.readout_ceiling \\"
echo "      --data ${DATA_DIR}/val.lmdb --template-data ${DATA_DIR}/train.lmdb \\"
echo "      --type-key centre-cn-ligands --cache-dir ${BLOCK_DIR}/cache --workers ${WORKERS} \\"
echo "      --json-out ${REPORT_DIR}/readout_g1_fine.json"
echo "  ${PYTHON} -m cgfm.scripts.check_readout_gates \\"
echo "      --coarse ${REPORT_DIR}/readout_g1_coarse.json \\"
echo "      --fine ${REPORT_DIR}/readout_g1_fine.json \\"
echo "      --stamp ${BLOCK_DIR}/phase0_passed.json"
echo
echo "Preparation finished. Block tables are in ${BLOCK_DIR}."

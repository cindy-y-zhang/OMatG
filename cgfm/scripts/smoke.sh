#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline on a few dozen structures.
#
#   cgfm/scripts/smoke.sh
#
# Builds a tiny split, precomputes both fixed partitions on it, trains all four arms for two epochs, then generates
# several draws for one arm, scores them and collects the table. Nothing it produces is a result; the point is to catch
# configuration and plumbing errors before an expensive run. Takes a few minutes on a laptop CPU.
#
# Run from the repository root.

set -euo pipefail

DATA_DIR="cgfm/smoke_data"
GROUP_DIR="cgfm/smoke_groups"
RUN_ROOT="cgfm/smoke_runs"
CONFIGS="cgfm/configs"

echo "== building a tiny split =="
python -m cgfm.scripts.make_subset --data omg/data/mpts_52/train.lmdb --out "${DATA_DIR}/train.lmdb" --count 64
python -m cgfm.scripts.make_subset --data omg/data/mpts_52/val.lmdb   --out "${DATA_DIR}/val.lmdb"   --count 16
python -m cgfm.scripts.make_subset --data omg/data/mpts_52/test.lmdb  --out "${DATA_DIR}/test.lmdb"  --count 16

echo
echo "== precomputing partitions =="
for split in train val; do
    python -m cgfm.scripts.precompute_groups --data "${DATA_DIR}/${split}.lmdb" --out-dir "${GROUP_DIR}" --workers 4
done

echo
echo "== training every arm for two epochs =="
for arm in atomwise kmedoids shells learned; do
    case "${arm}" in
        shells) method=shells ;;
        *)      method=kmedoids ;;
    esac
    echo "-- ${arm} --"
    rm -rf "${RUN_ROOT}/${arm}"
    mkdir -p "${RUN_ROOT}/${arm}/seed0"
    python -m cgfm.main fit \
        --config "${CONFIGS}/base_mpts52.yaml" \
        --config "${CONFIGS}/arm_${arm}.yaml" \
        --config "${CONFIGS}/smoke.yaml" \
        --data.train_group_file "${GROUP_DIR}/train.${method}.npz" \
        --data.val_group_file "${GROUP_DIR}/val.${method}.npz" \
        --model.si.init_args.integration_time_steps 5 \
        --model.si.init_args.diagnostics_every 1 \
        --trainer.default_root_dir "${RUN_ROOT}/${arm}/seed0" \
        --trainer.logger.init_args.save_dir "${RUN_ROOT}/${arm}/seed0"
done

echo
echo "== checking the collapse diagnostics were logged =="
# These are the whole basis for telling a genuine learned partition from collapse onto geometric clustering, and they
# are easy to lose silently: they are recorded inside the interpolant and only surface if the callback is wired up.
LEARNED_METRICS=$(find "${RUN_ROOT}/learned" -name metrics.csv | head -1)
for column in cg_ari_vs_kmedoids cg_ari_vs_shells cg_fine_energy_fraction cg_temperature; do
    if ! head -1 "${LEARNED_METRICS}" | tr ',' '\n' | grep -qx "${column}"; then
        echo "diagnostics missing: ${column} was never logged to ${LEARNED_METRICS}" >&2
        exit 1
    fi
done
echo "all four collapse diagnostics present in ${LEARNED_METRICS}"

echo
echo "== generating two draws and scoring them =="
# Two draws rather than the five used in a real evaluation, which is enough to exercise the best-of-n path.
EVAL_DIR="${RUN_ROOT}/shells/seed0/eval/nfe5"
mkdir -p "${EVAL_DIR}"
DRAWS=()
for draw in 0 1; do
    DRAWS+=("${EVAL_DIR}/draw${draw}.xyz")
    python -m cgfm.main predict \
        --config "${CONFIGS}/base_mpts52.yaml" \
        --config "${CONFIGS}/arm_shells.yaml" \
        --config "${CONFIGS}/smoke.yaml" \
        --seed_everything "${draw}" \
        --data.train_group_file "${GROUP_DIR}/train.shells.npz" \
        --data.val_group_file "${GROUP_DIR}/val.shells.npz" \
        --model.si.init_args.integration_time_steps 5 \
        --model.generation_xyz_filename "${EVAL_DIR}/draw${draw}.xyz"
done

python -m cgfm.scripts.score \
    --generated "${DRAWS[@]}" \
    --reference "${DATA_DIR}/test.lmdb" \
    --out "${EVAL_DIR}/score.json" \
    --workers 4 \
    --label "smoke shells/seed0"

echo
echo "== collecting the table =="
python -m cgfm.scripts.collect_results --runs "${RUN_ROOT}" --source test --nfe 5
python -m cgfm.scripts.collect_results --runs "${RUN_ROOT}" --source validation

echo
echo "== smoke test finished =="

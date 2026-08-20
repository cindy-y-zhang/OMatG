#!/usr/bin/env bash
#
# Package everything the eight-A100 box needs for the direct-geometry sweep, and nothing it does not.
#
#   bash direct_geometry/scripts/package_bundle.sh                    # writes /tmp/direct_geometry_bundle.tar.zst
#   bash direct_geometry/scripts/package_bundle.sh /path/bundle.tar   # somewhere else
#
# WHY THIS REFUSES TO RUN WITHOUT A PASSING GATE
#
# The bundle exists to spend fifteen GPU-weeks. Every one of those weeks is justified by a chain of evidence -- DG0 that the
# features are finite, varying and affordable; DG1 and DG2 that they carry information a chemistry-only control does not;
# DG3 that an arm using them can optimise and does not regress. A bundle built without that chain is a bundle whose result
# cannot be attributed to anything, so the gate stamps are checked here and the build stops rather than warns.
#
# The stamps travel inside the bundle as well, because a number read on the far side against a paraphrase of the gates is
# a number read against a paraphrase. The remote launcher re-reads them before it will start.
#
# WHAT GOES IN
#
# Code, configs, tests, the three MPTS-52 splits, the hundred-structure smoke split, the gate stamps, the launcher, the
# report script, a checksum manifest and a verifier. Self-contained: no network access is needed on arrival beyond
# installing the environment.
#
# WHAT STAYS OUT
#
# Checkpoints and run outputs. They are gigabytes, they are outputs rather than inputs, and a stale checkpoint copied onto
# a fresh machine is how a "resumed" run silently continues from the wrong weights. The remote side trains from scratch,
# which is the only thing that makes its numbers comparable to each other.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TARGET="${1:-/tmp/direct_geometry_bundle.tar}"
DATA_DIR="${DG_DATA_DIR:-omg/data/mpts_52}"
SMOKE_DATA_DIR="${DG_SMOKE_DATA_DIR:-cgfm/overfit_data}"
# Small splits the staged tests read. Separate from SMOKE_DATA_DIR because they are not the same directory: the launchers'
# smoke and memorise modes train on the hundred-structure overfit split, and the audit tests sample from a different small
# sample of MPTS-52. Both are a few hundred kilobytes, and the staged test run below is what proves the list is complete.
IFS=',' read -r -a TEST_DATA_DIRS <<< "${DG_TEST_DATA_DIRS:-cgfm/smoke_data,cgfm/overfit_data}"
REPORTS="${DG_REPORT_DIR:-direct_geometry/reports}"
BASE_CONFIG="${DG_BASE_CONFIG:-cgfm/configs/atomwise_mpts52.yaml}"

if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
    echo "run this from the repository root" >&2
    exit 1
fi

if [[ -n "${DG_PYTHON:-}" ]]; then PYTHON="${DG_PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
else PYTHON="python3"
fi

echo "== checking the gates =="
# The per-gate semantics live in direct_geometry.scripts.gates rather than here. DG2's own verdict is a selection between
# radial and radial-plus-angular, not a bar, so a shell script reading a generic "passed" key would refuse to build a
# bundle precisely because a factor had been correctly dropped.
if ! "${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" --require dg0,probes,dg3; then
    if [[ "${DG_IGNORE_GATE:-0}" == "1" ]]; then
        echo
        echo "DG_IGNORE_GATE=1, so building an unauthorised bundle anyway." >&2
    else
        echo
        echo "This bundle exists to spend fifteen GPU-weeks, and every one of them is justified by the chain above." >&2
        echo "Build it after the gates pass, or set DG_IGNORE_GATE=1 to package a deliberate reproduction." >&2
        exit 1
    fi
fi
PROMOTED="$("${PYTHON}" -m direct_geometry.scripts.gates --reports "${REPORTS}" --promoted || echo unknown)"

echo
echo "== checking the inputs =="
for split in train val test; do
    [[ -s "${DATA_DIR}/${split}.lmdb" ]] || { echo "missing ${DATA_DIR}/${split}.lmdb" >&2; exit 1; }
    [[ -s "${SMOKE_DATA_DIR}/${split}.lmdb" ]] || { echo "missing ${SMOKE_DATA_DIR}/${split}.lmdb" >&2; exit 1; }
done
echo "  ${DATA_DIR} and ${SMOKE_DATA_DIR} present"

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT
BUNDLE="${STAGING}/direct_geometry_bundle"
mkdir -p "${BUNDLE}"

echo
echo "== staging the source =="
# By pattern rather than from git, because direct_geometry is new and an uncommitted file missing from the bundle would
# fail on import after the transfer rather than here. Run outputs and data are excluded and staged deliberately below.
find omg direct_geometry cgfm/configs -type f \
     \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' \) \
     -not -path '*/__pycache__/*' \
     -not -path 'direct_geometry/runs/*' \
     -not -path 'direct_geometry/a100_runs/*' \
     -not -path 'direct_geometry/bundles/*' \
     -not -path 'omg/data/*' \
    | while read -r path; do
        mkdir -p "${BUNDLE}/$(dirname "${path}")"
        cp "${path}" "${BUNDLE}/${path}"
    done
for path in pyproject.toml setup.py README.md; do
    [[ -f "${path}" ]] && cp "${path}" "${BUNDLE}/${path}"
done

# The bundle is useless without each of these, and a missing one must stop the build here rather than surface remotely
# after a transfer.
for path in "${BASE_CONFIG}" \
            direct_geometry/__init__.py direct_geometry/neighbors.py direct_geometry/features.py \
            direct_geometry/layers.py direct_geometry/encoder.py direct_geometry/batches.py \
            direct_geometry/configs/encoder.yaml direct_geometry/configs/a100.yaml \
            direct_geometry/configs/smoke.yaml direct_geometry/configs/local.yaml \
            direct_geometry/scripts/run_a100.sh direct_geometry/scripts/wandb_logging.sh \
            direct_geometry/scripts/evaluate.sh direct_geometry/scripts/evaluate.py \
            direct_geometry/scripts/a100_report.py direct_geometry/scripts/audit_geometry.py \
            direct_geometry/scripts/probe_features.py direct_geometry/scripts/target_mse.py \
            direct_geometry/scripts/local_gate.py \
            direct_geometry/scripts/gates.py direct_geometry/scripts/runs.py \
            direct_geometry/tests/conftest.py direct_geometry/tests/test_encoder.py \
            direct_geometry/tests/test_neighbors.py direct_geometry/tests/test_features.py \
            direct_geometry/tests/test_a100_report.py; do
    [[ -f "${BUNDLE}/${path}" ]] || { echo "missing ${path} from the staged tree" >&2; exit 1; }
done
echo "  $(find "${BUNDLE}" -name '*.py' | wc -l) python files, $(find "${BUNDLE}" -name '*.yaml' | wc -l) configs"

echo
echo "== staging the data =="
mkdir -p "${BUNDLE}/${DATA_DIR}" "${BUNDLE}/${SMOKE_DATA_DIR}"
cp "${DATA_DIR}"/{train,val,test}.lmdb "${BUNDLE}/${DATA_DIR}/"
cp "${SMOKE_DATA_DIR}"/{train,val,test}.lmdb "${BUNDLE}/${SMOKE_DATA_DIR}/"
for extra in "${TEST_DATA_DIRS[@]}"; do
    [[ -d "${extra}" ]] || { echo "missing ${extra}, which the staged tests read" >&2; exit 1; }
    mkdir -p "${BUNDLE}/${extra}"
    cp "${extra}"/*.lmdb "${BUNDLE}/${extra}/"
done
echo "  $(du -sh "${BUNDLE}/${DATA_DIR}" | cut -f1) of splits, $(du -sh "${BUNDLE}/${SMOKE_DATA_DIR}" | cut -f1) of smoke split"

echo
echo "== staging the evidence =="
mkdir -p "${BUNDLE}/${REPORTS}"
for path in "${REPORTS}"/*.json; do
    [[ -f "${path}" ]] && cp "${path}" "${BUNDLE}/${REPORTS}/"
done
echo "  $(find "${BUNDLE}/${REPORTS}" -name '*.json' | wc -l) gate stamps"

# The source state, so a result can be traced to the tree that produced it even after the tree has moved on. Recorded as
# data rather than as prose because the remote launcher reads it.
"${PYTHON}" - "${BUNDLE}" "${PROMOTED}" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path


def git(*arguments: str) -> str:
    try:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


bundle = Path(sys.argv[1])
(bundle / "PACKAGING.json").write_text(json.dumps({
    "built_at": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip(),
    "commit": git("rev-parse", "HEAD"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "dirty": bool(git("status", "--porcelain")),
    "describe": git("describe", "--always", "--dirty"),
    "promoted_descriptor": sys.argv[2],
    "built_on": {"platform": platform.platform(), "python": platform.python_version()},
}, indent=2, sort_keys=True) + "\n")
PY

cat > "${BUNDLE}/VERIFY.sh" <<'VERIFY'
#!/usr/bin/env bash
# Re-read every file and compare it against the manifest written when the bundle was built.
#
# Run this before launching anything. A partial transfer of an LMDB leaves a file that is present, plausibly sized, and
# whose every structure past the truncation is missing; the alternative to catching it here is discovering it in a match
# rate several days later.
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --quiet --check SHA256SUMS
echo "every file matches the manifest."
VERIFY
chmod +x "${BUNDLE}/VERIFY.sh"

cat > "${BUNDLE}/START-HERE.md" <<README
# Direct geometric features for atomwise crystal generation

Built $(date -u +"%Y-%m-%d %H:%M UTC") from $(git describe --always --dirty 2>/dev/null || echo "an untracked tree").
Promoted descriptor: \`${PROMOTED}\`.

## The question

Two earlier attempts in this project gave the denoiser a *semantic* variable -- a rigid block pose, then a
coordination-geometry token -- and both found the variable either unreadable by the encoder or worth less than the
optimisation capacity it consumed. The diagnosis that survived both is narrower: \`CSPNet\` never computes an interatomic
distance, and on a fully connected graph it cannot represent periodic image multiplicity at all.

So this drops the semantic variable and adds continuous, rotationally and translationally invariant summaries of each
atom's local environment straight into the node stream, plus a separately switchable corrected periodic message graph.
Nothing in \`omg\` is modified: the interpolants, losses, sampler and heads are the baseline's, so a direct-geometry run and
the atomwise baseline solve the same generative problem and their match rates are comparable.

## On arrival

    ./VERIFY.sh                                          # checksums. Do not skip.
    pip install -e .                                     # or point DG_PYTHON at an existing environment
    bash direct_geometry/scripts/run_a100.sh verify      # environment, gates, tests, dry run
    bash direct_geometry/scripts/run_a100.sh smoke       # all five arms, two epochs each, one card
    bash direct_geometry/scripts/run_a100.sh launch      # the sweep
    bash direct_geometry/scripts/run_a100.sh report      # paired contrasts and the verdict

The smoke step is not optional. It runs every arm end to end on 100 structures including a full sampling and match-rate
validation pass, and it is the only thing between a one-character configuration error and fifteen wasted runs.

\`launch\` refuses to start unless \`${REPORTS}/LOCAL-GATE.json\` records a passing Gate DG3. Set \`DG_IGNORE_GATE=1\`
only to reproduce a superseded launch deliberately.

## The design

Five arms, three shared seeds, fifteen runs of 1,600 epochs. Two binary factors crossed, plus one attribution arm:

| arm | \`message_graph\` | \`feature_mode\` | isolates |
| --- | --- | --- | --- |
| A | \`fc\` | \`none\` | nothing: the baseline code path with every addition zeroed |
| B | \`periodic_distance\` | \`none\` | periodic multiedges and true edge lengths, together |
| C | \`fc\` | \`${PROMOTED}\` | the node descriptor |
| D | \`periodic_distance\` | \`${PROMOTED}\` | both |
| E | \`fc_distance\` | \`none\` | the edge length channel alone, on the baseline's own topology |

Every arm is the same class with the same parameter count, and both additions enter through zero-initialised weights, so
all five are *functionally identical at initialisation*. Any divergence in a learning curve is something training found
rather than a different starting point; \`direct_geometry/tests/test_encoder.py\` asserts this directly.

Arm A is a re-run, not a quoted number. The project's 26.82 per cent came from stock \`CSPNetFull\`, and a cross-arm
contrast has to hold the code path fixed. Do not compare anything here against 26.82; compare arms against each other.

Arm E exists because B changes two things at once. The baseline edge network is fed a sinusoidal embedding of the
fractional offset alongside the lattice Gram matrix, so a Cartesian length is a bilinear form it must learn to construct;
this project's own auxiliary classifier gained 7.29 accuracy points when simply handed the distances. E supplies that
product on the unchanged topology, so E-A is the distance channel alone and B-E is the periodic topology net of it.
Without E, a B-A win is unattributable and the cheapest explanation goes untested.

## Reading the result

Contrasts are paired within seed and only then averaged. Seed-to-seed spread in MPTS-52 match rate is of the same order as
the effect being looked for, so an unpaired difference of arm means buries the effect inside the seed spread.

* **Deployment contrast.** D-A must clear both a practical +2 points and two paired standard errors.
* **Descriptor main effect.** Estimated from C-A and D-B. A direct-feature claim requires it positive at two paired
  standard errors. A D-A win caused only by the graph is reported as graph-only and is *not* evidence for the adapter.
* **Graph effect.** Estimated from B-A and D-C, and decomposed with E: E-A is the length channel, B-E the topology.

Two spreads are reported and never pooled: the seed standard error, which is uncertainty about the arm and is what the
verdicts are read on, and the within-checkpoint spread over five repeated validation draws, which is a property of the
sampler. Architecture and checkpoint are selected on validation; the test split is scored once afterwards at 50 and 210
NFE without retuning.

A failed descriptor main effect ends the node-adapter program rather than triggering an unregistered architecture search.

## What the local evidence already says, in \`${REPORTS}/\`

* **DG0** -- the descriptor's channels are finite and varying across the whole path, the periodic graph really does carry
  image multiplicity, and every arm is inside a 30 per cent cost ceiling against arm A.
* **DG1** -- at the clean endpoint the radial descriptor adds 26.1 coordination-number accuracy points and 0.81 bits over
  a chemistry-only floor of 34.6 per cent. Radial features were promoted.
* **DG2** -- angular features added 2.7 shape points against a required 5, so they were dropped. Hence
  \`feature_mode: ${PROMOTED}\`.
* **DG3** -- the local screen. Screening evidence only: the baseline needs 1,200 epochs to reach 26.82 per cent, so a
  200-epoch number is a point on a rising curve.

One caveat worth carrying: DG1 measures the descriptor of a *clean* structure. Measured along the path, the same
descriptor is worth +26.1 points at t=1 but only +0.7 at t=0.5, because a descriptor of a noisy state is a descriptor of
noise. Arms C and D therefore have most of their information available only near the end of the trajectory. That is a real
limit of this design and the reason a jointly generated descriptor variable is the natural follow-up.

## Practical notes

On an 80 GB A100 the configured batch is 128 with accumulation 8, preserving the effective batch of 1,024 the learning
rate was tuned at, at \`32-true\` precision. Fifteen runs over eight cards is scheduled as a work queue rather than fixed
waves, so no card idles while another finishes.

Set \`DG_WORKERS\` to roughly a sixth of the core count. Validation structure-matches 5,000 structures across worker
processes and several runs share a validation schedule, so the default can mean dozens of processes competing whenever
those schedules coincide.

Weights & Biases logging goes to project \`geo-adapter\` when credentials are present and falls back to offline files
otherwise; \`metrics.csv\` is always written per run and is what the report reads.
README

# The staged tree runs its own tests before it is sealed. This is the check that makes the hand-written required-file list
# above a convenience rather than the guarantee: a missing module, a missing small split or a path that only resolves in the
# source checkout all fail here, at build time, instead of on arrival after a transfer. It costs a few seconds.
#
# Run with the staged tree as the working directory and on the path, so nothing can resolve back into the repository it was
# copied from. That is the whole point: a test that passes only because the original tree is still reachable proves nothing
# about the bundle.
echo
echo "== proving the staged tree stands alone =="
if [[ "${DG_SKIP_STAGED_TESTS:-0}" == "1" ]]; then
    echo "  skipped by DG_SKIP_STAGED_TESTS=1"
else
    STAGED_PYTHON="${PYTHON}"
    [[ "${STAGED_PYTHON}" == /* ]] || STAGED_PYTHON="${PWD}/${STAGED_PYTHON}"
    if ! ( cd "${BUNDLE}" && PYTHONPATH="${BUNDLE}" "${STAGED_PYTHON}" -m pytest direct_geometry/tests -q \
               > "${STAGING}/tests.log" 2>&1 ); then
        echo
        echo "the staged tree fails its own tests, so the bundle is not self-contained:" >&2
        tail -25 "${STAGING}/tests.log" >&2
        echo >&2
        echo "This is what the check is for. Add whatever the failures name to the staging step above; do not ship a" >&2
        echo "bundle that only works because the source checkout was still reachable." >&2
        exit 1
    fi
    echo "  $(grep -Eo '[0-9]+ passed' "${STAGING}/tests.log" | tail -1) in the staged tree"
fi

# Last, so that everything the bundle ships is in the manifest, this file included. Written any earlier and the files
# generated after it are either unchecked or recorded with the checksum of whatever they replaced.
echo
echo "== writing the checksum manifest =="
( cd "${BUNDLE}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "  $(wc -l < "${BUNDLE}/SHA256SUMS") files"

echo
echo "== building the archive =="
mkdir -p "$(dirname "${TARGET}")"
if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="${TARGET%.tar}.tar.zst"
    tar -C "${STAGING}" -cf - direct_geometry_bundle | zstd -19 -T0 -q -o "${ARCHIVE}" -f
else
    ARCHIVE="${TARGET%.tar}.tar.gz"
    tar -C "${STAGING}" -czf "${ARCHIVE}" direct_geometry_bundle
fi

echo
echo "wrote ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1)), $(find "${BUNDLE}" -type f | wc -l) files"
echo
echo "copy it over and unpack:"
echo "  rsync -avP --partial ${ARCHIVE} A100BOX:/scratch/"
echo "  ssh A100BOX 'cd /scratch && tar --zstd -xf $(basename "${ARCHIVE}") 2>/dev/null || tar -xzf $(basename "${ARCHIVE}")'"
echo "  ssh A100BOX 'cd /scratch/direct_geometry_bundle && ./VERIFY.sh'"
echo
echo "then read START-HERE.md, which has the launch commands."

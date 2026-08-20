"""
Reading finished runs, and the paired arithmetic every verdict in this project is stated in.

Shared by the local gate and the A100 report rather than written twice. Two copies of "which match rate is read" or of
"how a paired difference is formed" would eventually disagree, and the disagreement would surface as two reports that
cannot both be right about the same runs.

WHICH MATCH RATE IS READ

The best validation match rate over the run, with the final one recorded beside it. The best is the checkpoint downstream
evaluation actually loads, so gating on anything else would grade a model nobody uses. It is biased upwards as the maximum
of several noisy draws, but every arm has an identical validation cadence and an identical number of draws, so the bias is
common to both sides of every contrast and largely cancels in the difference.

WHY EVERYTHING IS PAIRED WITHIN SEED

Seed-to-seed spread in MPTS-52 match rate is of the same order as the effect being looked for. A difference of arm means
buries the effect inside that spread; a mean of within-seed differences does not, because the seed is held fixed on both
sides. So the seed is the unit of replication throughout, and a seed missing either arm is reported as unpaired rather
than quietly averaged in.
"""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Optional


MSE_IMPROVEMENT = 0.02
"""
Default fractional improvement in denoising error counted as an improvement in one time bucket.

Match rate is a thresholded, sampled, noisy end-to-end score; the denoising error is what the model is actually trained to
reduce and is far quieter. Requiring both guards against a match-rate difference that is sampling luck.
"""


def digest(path: Path) -> str:
    """
    Return the SHA-256 of one file.

    :param path:
        File to hash.
    :type path: Path

    :return:
        Hex digest.
    :rtype: str
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def tree_digest(root: Path, patterns: tuple[str, ...] = ("*.py", "*.yaml", "*.sh")) -> dict[str, str]:
    """
    Return the SHA-256 of every matching file under a path, keyed by path relative to the repository.

    Recorded per file rather than as one rolled-up hash, so that a mismatch says which file moved instead of only that
    something did.

    :param root:
        Directory or file to walk.
    :type root: Path
    :param patterns:
        Filename patterns to include.
        Defaults to Python, YAML and shell sources.
    :type patterns: tuple[str, ...]

    :return:
        Digests keyed by path.
    :rtype: dict[str, str]
    """
    if root.is_file():
        return {str(root): digest(root)}
    found: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if "__pycache__" in path.parts or path.name.startswith("."):
                continue
            found[str(path)] = digest(path)
    return found


def commit() -> dict[str, str]:
    """
    Return the current commit and whether the tree is dirty.

    A dirty tree is recorded rather than refused: these are research runs and the working tree is usually ahead of the
    last commit. Recording it is what makes the stamp honest.

    :return:
        Commit hash and dirty flag.
    :rtype: dict[str, str]
    """
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return {"commit": revision.strip(), "dirty": str(bool(status.strip())).lower()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": "unknown"}


def read_metrics(run_dir: Path) -> Optional[dict]:
    """
    Read one run's match-rate history out of the CSV logger's output.

    :param run_dir:
        Directory of the run.
    :type run_dir: Path

    :return:
        Best and final match rate, the epoch each was reached at, the number of validations, and the file's digest, or
        None if the run has no readable metrics.
    :rtype: Optional[dict]
    """
    path = run_dir / "metrics.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    rates: list[tuple[float, float]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            value, epoch = row.get("match_rate"), row.get("epoch")
            if value not in (None, "") and epoch not in (None, ""):
                rates.append((float(epoch), float(value)))
    if not rates:
        return None
    best_epoch, best = max(rates, key=lambda pair: pair[1])
    final_epoch, final = rates[-1]
    return {"best_match_rate": best, "best_epoch": best_epoch, "final_match_rate": final,
            "final_epoch": final_epoch, "validations": len(rates), "complete": (run_dir / "COMPLETE").is_file(),
            "metrics_sha256": digest(path)}


def read_target_mse(run_dir: Path) -> Optional[dict]:
    """
    Read one run's fixed-time denoising errors, if they have been measured.

    :param run_dir:
        Directory of the run.
    :type run_dir: Path

    :return:
        Errors keyed by time label and field, plus the report's digest, or None if not measured.
    :rtype: Optional[dict]
    """
    path = run_dir / "TARGET-MSE.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("identity_failures"):
        return {"errors": None, "identity_failures": report["identity_failures"],
                "target_mse_sha256": digest(path)}
    errors = {label: {field: entry[field]["mean_squared_error"] for field in ("pos", "cell")}
              for label, entry in report["errors"].items()}
    return {"errors": errors, "identity_failures": [], "target_mse_sha256": digest(path)}


def read_sampling(run_dir: Path) -> Optional[dict]:
    """
    Read one run's repeated-draw and locked-test scores, if they have been measured.

    Kept apart from the training-time match rate on purpose. The spread over repeated draws of one checkpoint is a
    property of the sampler, and the spread over seeds is uncertainty about the arm; pooling them would understate the
    second by diluting it with the first.

    :param run_dir:
        Directory of the run.
    :type run_dir: Path

    :return:
        Repeated validation draws and locked test scores keyed by sampling budget, or None if neither was measured.
    :rtype: Optional[dict]
    """
    found: dict[str, dict] = {}
    for path in sorted(run_dir.glob("eval/*/SCORE.json")):
        report = json.loads(path.read_text())
        found[f"{report['split']}_nfe{report['integration_time_steps']}"] = {
            "match_rates": report["match_rates"], "mean": report["mean_match_rate"],
            "standard_deviation": report["standard_deviation"], "draws": len(report["match_rates"]),
            "checkpoint": report.get("checkpoint"), "score_sha256": digest(path)}
    return found or None


def collect(run_root: Path, modes: Optional[tuple[str, ...]] = None) -> dict[str, dict[str, dict]]:
    """
    Read runs under the run root, grouped by mode and then by run name.

    Restricting the modes is what keeps a superseded result from being read as a current one. When the definition of a
    check changes, its old runs are answers to a different question; they are worth keeping as evidence for why the check
    changed, and they must not reach a gate. Naming the modes a reader wants makes both true at once, and leaves the
    unread directories visible to :func:`unread_modes` rather than silently absorbed.

    :param run_root:
        Directory the launcher writes its runs into.
    :type run_root: Path
    :param modes:
        Mode directories to read, or None to read every one present.
    :type modes: Optional[tuple[str, ...]]

    :return:
        Runs keyed by mode and run name.
    :rtype: dict[str, dict[str, dict]]
    """
    runs: dict[str, dict[str, dict]] = {}
    if not run_root.is_dir():
        return runs
    for mode_dir in sorted(p for p in run_root.iterdir()
                           if p.is_dir() and (modes is None or p.name in modes)):
        for run_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
            metrics = read_metrics(run_dir)
            if metrics is None:
                continue
            entry = dict(metrics)
            entry["path"] = str(run_dir)
            mse = read_target_mse(run_dir)
            if mse is not None:
                entry.update(mse)
            sampling = read_sampling(run_dir)
            if sampling is not None:
                entry["sampling"] = sampling
            runs.setdefault(mode_dir.name, {})[run_dir.name] = entry
    return runs


def unread_modes(run_root: Path, modes: tuple[str, ...]) -> list[str]:
    """
    Return the mode directories present under the run root that a reader is not reading.

    Reported rather than ignored. A directory of finished runs sitting beside the ones a gate reads is exactly what a
    person scanning a stamp would mistake for the gate's input, so the stamp says which directories it declined and the
    mistake becomes impossible instead of merely unlikely.

    :param run_root:
        Directory the launcher writes its runs into.
    :type run_root: Path
    :param modes:
        Modes that are being read.
    :type modes: tuple[str, ...]

    :return:
        Names of the directories holding runs that were not read.
    :rtype: list[str]
    """
    if not run_root.is_dir():
        return []
    return sorted(path.name for path in run_root.iterdir()
                  if path.is_dir() and path.name not in modes and any(path.glob("*/metrics.csv")))


def arm_of(name: str) -> str:
    """
    Return the arm letter of a run directory name such as ``D_seed1``.

    :param name:
        Run directory name.
    :type name: str

    :return:
        Arm letter.
    :rtype: str
    """
    return name.split("_", 1)[0]


def seed_of(name: str) -> str:
    """
    Return the seed of a run directory name such as ``D_seed1``.

    :param name:
        Run directory name.
    :type name: str

    :return:
        Seed as a string, or "" if the name carries none.
    :rtype: str
    """
    _, _, suffix = name.partition("_seed")
    return suffix


def spread(values: list[float]) -> dict[str, Optional[float]]:
    """
    Return the mean, sample standard deviation and standard error of a list of numbers.

    The standard deviation is the sample one, with the Bessel correction, because these are three seeds drawn from the
    population of seeds and not the population itself. With three of them the estimate is weak, which is a property of
    the design and is why it is reported rather than only used.

    :param values:
        The numbers.
    :type values: list[float]

    :return:
        Mean, sample standard deviation, standard error and count. The spreads are None below two values.
    :rtype: dict[str, Optional[float]]
    """
    count = len(values)
    if count == 0:
        return {"mean": None, "standard_deviation": None, "standard_error": None, "count": 0}
    mean = sum(values) / count
    if count < 2:
        return {"mean": mean, "standard_deviation": None, "standard_error": None, "count": count}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    deviation = math.sqrt(variance)
    return {"mean": mean, "standard_deviation": deviation, "standard_error": deviation / math.sqrt(count),
            "count": count}


def match_contrast(runs: dict[str, dict], better: str, worse: str) -> dict:
    """
    Return the paired match-rate difference between two arms, in points, seed by seed.

    :param runs:
        Runs of one mode, keyed by run name.
    :type runs: dict[str, dict]
    :param better:
        Arm expected to be ahead.
    :type better: str
    :param worse:
        Arm it is compared against.
    :type worse: str

    :return:
        Per-seed differences in points, their mean and spread, and the seeds that were missing a side.
    :rtype: dict
    """
    left = {seed_of(name): entry for name, entry in runs.items() if arm_of(name) == better}
    right = {seed_of(name): entry for name, entry in runs.items() if arm_of(name) == worse}
    shared = sorted(set(left) & set(right))
    points = {seed: 100.0 * (left[seed]["best_match_rate"] - right[seed]["best_match_rate"]) for seed in shared}
    statistics = spread(list(points.values()))
    unpaired = sorted((set(left) | set(right)) - set(shared))
    return {"contrast": f"{better}-{worse}", "points": points, "mean_points": statistics["mean"],
            "standard_error": statistics["standard_error"],
            "standard_deviation": statistics["standard_deviation"],
            "seeds": shared, "unpaired_seeds": unpaired}


def error_contrast(runs: dict[str, dict], better: str, worse: str, field: str,
                   improvement: float = MSE_IMPROVEMENT) -> dict:
    """
    Return the paired fractional change in denoising error between two arms, per time bucket.

    Negative is an improvement, because the quantity is an error. Averaged over seeds within each bucket, so a bucket
    where one seed is missing is reported from the seeds that have it rather than dropped.

    :param runs:
        Runs of one mode, keyed by run name.
    :type runs: dict[str, dict]
    :param better:
        Arm expected to be ahead.
    :type better: str
    :param worse:
        Arm it is compared against.
    :type worse: str
    :param field:
        Which field's error, ``pos`` or ``cell``.
    :type field: str
    :param improvement:
        Fractional improvement counted as an improvement in one bucket.
        Defaults to :const:`MSE_IMPROVEMENT`.
    :type improvement: float

    :return:
        Fractional change per time bucket, the count that improved by the required margin, and the buckets available.
    :rtype: dict
    """
    left = {seed_of(name): entry for name, entry in runs.items()
            if arm_of(name) == better and entry.get("errors")}
    right = {seed_of(name): entry for name, entry in runs.items()
             if arm_of(name) == worse and entry.get("errors")}
    shared = sorted(set(left) & set(right))
    buckets: dict[str, float] = {}
    for seed in shared:
        for label, values in left[seed]["errors"].items():
            reference = right[seed]["errors"].get(label, {}).get(field)
            if reference in (None, 0.0):
                continue
            buckets.setdefault(label, 0.0)
            buckets[label] += (values[field] - reference) / reference / len(shared)
    return {"contrast": f"{better}-{worse}", "field": field, "fractional_change": buckets,
            "improved_buckets": sum(1 for value in buckets.values() if value <= -improvement),
            "total_buckets": len(buckets), "seeds": shared}


def average_contrasts(contrasts: list[dict], name: str) -> dict:
    """
    Combine several estimates of the same effect into one, pooling their per-seed differences.

    A main effect in a crossed design is estimated at both levels of the other factor: the descriptor effect appears as
    C-A with the fully connected graph and as D-B with the periodic one. Pooling the per-seed differences rather than
    averaging the two means keeps the seed as the unit of replication, so the standard error still reflects seed spread
    and not the spread between two arbitrarily ordered estimates.

    :param contrasts:
        The contrasts to pool, as returned by :func:`match_contrast`.
    :type contrasts: list[dict]
    :param name:
        Name for the pooled effect.
    :type name: str

    :return:
        The pooled differences, their mean and spread, and which contrasts went in.
    :rtype: dict
    """
    pooled = [value for contrast in contrasts for value in contrast["points"].values()]
    statistics = spread(pooled)
    return {"effect": name, "from": [contrast["contrast"] for contrast in contrasts],
            "differences": pooled, "mean_points": statistics["mean"],
            "standard_error": statistics["standard_error"],
            "standard_deviation": statistics["standard_deviation"], "count": statistics["count"]}


def clears(effect: dict, points: float, sigmas: float) -> Optional[bool]:
    """
    Whether an effect clears both a practical margin and a multiple of its own standard error.

    Both, because either alone is readable the wrong way. A difference that is large but within the noise is not a
    finding, and one that is outside the noise but a fraction of a point is not worth deploying.

    Strict positivity is required on top of the margin, and is not redundant with it. A margin of zero is deliberate for
    the descriptor main effect, whose practical bar is carried by the deployment contrast; without strict positivity,
    that zero margin would make the first condition vacuous. The second condition is vacuous in the same situation, since
    an effect of exactly zero also has a standard error of exactly zero when every seed agrees on it. So an arm that
    changed nothing at all would clear both bars and be reported as a win, which is the single worst failure this report
    could have.

    :param effect:
        A contrast or pooled effect, carrying ``mean_points`` and ``standard_error``.
    :type effect: dict
    :param points:
        Practical margin in match-rate points.
    :type points: float
    :param sigmas:
        Multiple of the standard error the mean must exceed.
    :type sigmas: float

    :return:
        True or False, or None if there were too few seeds to form a standard error.
    :rtype: Optional[bool]
    """
    mean, error = effect.get("mean_points"), effect.get("standard_error")
    if mean is None or error is None:
        return None
    return bool(mean > 0.0 and mean >= points and mean >= sigmas * error)

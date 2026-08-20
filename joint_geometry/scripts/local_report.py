"""Aggregate paired local-screen outcomes and apply preregistered stop gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


CONTRASTS = ("D", "P", "H", "R")
EVIDENCE_FILES = {
    "descriptor": "DESCRIPTOR-GATE.json",
    "backbone": "BACKBONE-GATE.json",
    "geometry_weight": "GEOMETRY-WEIGHT.json",
}


def read_outcomes(root: Path) -> dict[tuple[str, int, int], dict]:
    reports = {}
    for path in sorted(root.rglob("OUTCOMES*.json")):
        report = json.loads(path.read_text())
        key = (report["arm"], int(report["seed"]), int(report.get("draw", 0)))
        if key in reports:
            raise ValueError(f"Duplicate outcome report for {key}: {path}.")
        report["path"] = str(path)
        reports[key] = report
    return reports


def evidence_summary(root: Path) -> dict:
    """Bind model-screen conclusions to their frozen upstream decisions."""
    summary = {}
    for label, filename in EVIDENCE_FILES.items():
        path = root / filename
        if not path.is_file():
            summary[label] = None
            continue
        payload = json.loads(path.read_text())
        entry = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if label == "descriptor":
            entry["passed"] = payload["verdict"]["passed"]
            entry["promoted"] = payload["verdict"]["promoted"]
        elif label == "backbone":
            entry["passed"] = payload["verdict"]["passed"]
            entry["selected_backbone"] = payload["verdict"]["selected_backbone"]
        else:
            entry["geometry_weight"] = payload["geometry_weight"]
        summary[label] = entry
    return summary


def paired_vectors(first: dict, second: dict) -> tuple[np.ndarray, np.ndarray]:
    if first["num_compositions"] != second["num_compositions"]:
        raise ValueError("Paired outcome reports use different composition counts.")
    first_indices = [entry["index"] for entry in first["outcomes"]]
    second_indices = [entry["index"] for entry in second["outcomes"]]
    if first_indices != second_indices:
        raise ValueError("Paired outcome reports use different structure ordering.")
    left = np.asarray([entry["match"] for entry in first["outcomes"]], dtype=np.float64)
    right = np.asarray([entry["match"] for entry in second["outcomes"]], dtype=np.float64)
    return left, right


def contrast(
    reports: dict[tuple[str, int, int], dict],
    better: str,
    worse: str,
    bootstrap_draws: int,
    seed: int,
) -> dict | None:
    keys = sorted(
        (run_seed, draw)
        for arm, run_seed, draw in reports
        if arm == better and (worse, run_seed, draw) in reports
    )
    if not keys:
        return None
    differences, discordant_better, discordant_worse = [], 0, 0
    differences_by_seed: dict[int, list[float]] = {}
    vectors = []
    for run_seed, draw in keys:
        left, right = paired_vectors(
            reports[(better, run_seed, draw)], reports[(worse, run_seed, draw)]
        )
        difference = left - right
        vectors.append(difference)
        difference_points = 100.0 * float(difference.mean())
        differences.append(difference_points)
        differences_by_seed.setdefault(run_seed, []).append(difference_points)
        discordant_better += int(np.sum((left == 1) & (right == 0)))
        discordant_worse += int(np.sum((left == 0) & (right == 1)))

    generator = np.random.default_rng(seed)
    bootstrapped = np.empty(bootstrap_draws)
    for index in range(bootstrap_draws):
        sampled = []
        for vector in vectors:
            rows = generator.integers(0, len(vector), len(vector))
            sampled.append(vector[rows])
        bootstrapped[index] = 100.0 * float(np.concatenate(sampled).mean())
    discordant = discordant_better + discordant_worse
    p_value = (
        float(binomtest(discordant_better, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "better": better,
        "worse": worse,
        "paired_seed_draws": [f"seed{run_seed}_draw{draw}" for run_seed, draw in keys],
        "difference_points": float(np.mean(differences)),
        "seed_draw_differences_points": differences,
        "seed_differences_points": {
            str(run_seed): float(np.mean(values))
            for run_seed, values in sorted(differences_by_seed.items())
        },
        "bootstrap_ci95": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "mcnemar": {
            "better_only": discordant_better,
            "worse_only": discordant_worse,
            "two_sided_p": p_value,
        },
    }


def final_metric(path: Path, name: str) -> float | None:
    metrics = path / "metrics.csv"
    if not metrics.is_file():
        return None
    found = []
    with metrics.open() as handle:
        for row in csv.DictReader(handle):
            value = row.get(name)
            if value not in (None, ""):
                found.append(float(value))
    return found[-1] if found else None


def endpoint_consistency(run_root: Path, arm: str, seeds: list[int]) -> float | None:
    values = []
    for seed in seeds:
        evaluation_root = run_root / f"{arm}_seed{seed}" / "eval"
        for draw in sorted(evaluation_root.glob("draw*")):
            values.append(final_metric(draw, "val_geometry_endpoint_consistency"))
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def resource_reports(run_root: Path) -> dict:
    found = {}
    for run in sorted(path for path in run_root.iterdir() if path.is_dir()) if run_root.is_dir() else []:
        entry = {}
        for label, path in (
            ("training", run / "RESOURCE-TRAIN.json"),
            ("evaluation", run / "eval" / "draw0" / "RESOURCE-EVAL.json"),
        ):
            if path.is_file():
                entry[label] = json.loads(path.read_text())
        if entry:
            found[run.name] = entry
    return found


def outcome_metrics(reports: dict[tuple[str, int, int], dict]) -> dict[str, dict]:
    """Aggregate endpoint safeguards over every composition, seed, and draw."""
    by_arm: dict[str, list[dict]] = {}
    for (arm, _, _), report in reports.items():
        by_arm.setdefault(arm, []).extend(report["outcomes"])
    metrics = {}
    for arm, outcomes in sorted(by_arm.items()):
        matched_rmsds = [
            float(outcome["rmsd"])
            for outcome in outcomes
            if outcome.get("rmsd") is not None
        ]
        entry = {
            "compositions": len(outcomes),
            "match_rate": float(np.mean([outcome["match"] for outcome in outcomes])),
            "rmsd_on_matches": (
                float(np.mean(matched_rmsds)) if matched_rmsds else None
            ),
        }
        for name in ("position_target_mse", "cell_target_mse"):
            values = [
                float(outcome[name]) for outcome in outcomes if name in outcome
            ]
            entry[name] = float(np.mean(values)) if values else None
        metrics[arm] = entry
    return metrics


def sampler_jitter(reports: dict[tuple[str, int, int], dict]) -> dict[str, dict]:
    """Summarise repeated paired draws without treating them as new structures."""
    by_checkpoint: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for (arm, run_seed, draw), report in reports.items():
        rate = float(np.mean([outcome["match"] for outcome in report["outcomes"]]))
        by_checkpoint.setdefault((arm, run_seed), []).append((draw, rate))
    summary = {}
    for (arm, run_seed), entries in sorted(by_checkpoint.items()):
        entries.sort()
        rates = np.asarray([rate for _, rate in entries], dtype=np.float64)
        summary[f"{arm}_seed{run_seed}"] = {
            "draws": [draw for draw, _ in entries],
            "match_rates": rates.tolist(),
            "mean": float(rates.mean()),
            "std": float(rates.std()),
            "range": float(rates.max() - rates.min()),
        }
    return summary


def oracle_safeguards(metrics: dict[str, dict]) -> dict:
    comparisons = {}
    for name in ("position_target_mse", "cell_target_mse"):
        oracle_value = metrics.get("O", {}).get(name)
        baseline_value = metrics.get("D", {}).get(name)
        comparisons[name] = {
            "O": oracle_value,
            "D": baseline_value,
            "not_worse": (
                oracle_value is not None
                and baseline_value is not None
                and oracle_value <= baseline_value
            ),
        }
    return {
        "metrics": comparisons,
        "passed": all(item["not_worse"] for item in comparisons.values()),
    }


def promotion_failures(
    contrasts: dict[str, dict | None],
    paired_seeds: list[int],
    consistency_improvement: float | None,
) -> list[str]:
    """Return every unmet preregistered A100-promotion requirement."""
    failures = []
    required = ("D", "P", "H", "R")
    missing = [arm for arm in required if contrasts.get(arm) is None]
    if missing:
        failures.append(f"missing paired contrasts for {missing}")
        return failures

    j_d = contrasts["D"]
    j_p = contrasts["P"]
    assert j_d is not None and j_p is not None
    for arm, comparison in (("D", j_d), ("P", j_p)):
        if comparison["difference_points"] < 1.5:
            failures.append(f"J-{arm} is below 1.5 match points")
        if comparison["bootstrap_ci95"][0] <= 0.0:
            failures.append(f"J-{arm} paired-bootstrap interval is not above zero")
    for arm in ("H", "R"):
        comparison = contrasts[arm]
        assert comparison is not None
        if comparison["difference_points"] <= 0.0:
            failures.append(f"J is not directionally above {arm}")
    if len(paired_seeds) < 3:
        failures.append("fewer than three paired seeds are present")
    if any(value <= 0.0 for value in j_d["seed_differences_points"].values()):
        failures.append("not every paired seed has J above D")
    if any(value <= 0.0 for value in j_p["seed_differences_points"].values()):
        failures.append("not every paired seed has J above P")
    if consistency_improvement is None:
        failures.append("endpoint descriptor consistency is unavailable")
    elif consistency_improvement < 0.25:
        failures.append("endpoint descriptor consistency improves by less than 25% over P")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="joint_geometry/runs")
    parser.add_argument("--out", default="joint_geometry/reports/LOCAL-GATE.json")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--evidence-root", default="joint_geometry/reports")
    arguments = parser.parse_args()
    run_root = Path(arguments.run_root)
    reports = read_outcomes(run_root)
    gate_reports = {
        key: report for key, report in reports.items() if key[2] == 0
    }
    contrasts = {
        worse: contrast(
            gate_reports,
            "J",
            worse,
            arguments.bootstrap_draws,
            arguments.seed + index,
        )
        for index, worse in enumerate(CONTRASTS)
    }
    oracle = contrast(
        gate_reports,
        "O",
        "D",
        arguments.bootstrap_draws,
        arguments.seed + 100,
    )
    paired_seeds = sorted(
        {
            run_seed
            for arm, run_seed, draw in gate_reports
            if arm == "J" and draw == 0 and ("D", run_seed, draw) in reports
        }
    )
    j_consistency = endpoint_consistency(run_root, "J", paired_seeds)
    p_consistency = endpoint_consistency(run_root, "P", paired_seeds)
    safeguards = outcome_metrics(gate_reports)
    oracle_error_gate = oracle_safeguards(safeguards)
    oracle_passed = (
        oracle is not None
        and oracle["difference_points"] >= 2.0
        and oracle_error_gate["passed"]
    )
    consistency_improvement = (
        (p_consistency - j_consistency) / p_consistency
        if j_consistency is not None and p_consistency not in (None, 0.0)
        else None
    )

    complete = all(contrasts[name] is not None for name in CONTRASTS)
    failures = promotion_failures(
        contrasts,
        paired_seeds,
        consistency_improvement,
    )
    promoted = not failures
    screen_present = any(arm == "J" for arm, _, _ in gate_reports)
    if oracle is not None and not screen_present:
        if oracle_passed:
            protocol_status = "oracle_passed_screen_pending"
        else:
            protocol_status = "stopped_after_oracle"
            complete = True
            promoted = False
            failures = [
                "oracle did not improve match rate by at least 2 points "
                "without worsening position and cell errors"
            ]
    else:
        protocol_status = "screen_complete" if complete else "screen_incomplete"
    report = {
        "established_stock_match_rate": 0.2682,
        "paired_seeds": paired_seeds,
        "contrasts": contrasts,
        "oracle_contrast": oracle,
        "oracle_safeguards": oracle_error_gate,
        "oracle_passed": oracle_passed,
        "outcome_metrics": safeguards,
        "endpoint_consistency": {
            "J": j_consistency,
            "P": p_consistency,
            "relative_improvement": consistency_improvement,
        },
        "resources": resource_reports(run_root),
        "sampler_jitter": sampler_jitter(reports),
        "frozen_evidence": evidence_summary(Path(arguments.evidence_root)),
        "verdict": {
            "protocol_status": protocol_status,
            "complete": complete,
            "promote_to_a100": promoted,
            "passed": promoted,
            "failures": failures,
            "requirements": {
                "J_minus_D_points": 1.5,
                "J_minus_P_points": 1.5,
                "paired_bootstrap_ci_low_above_zero": True,
                "J_above_H_and_R": True,
                "three_seed_direction_agreement": True,
                "endpoint_consistency_improvement_over_P": 0.25,
            },
        },
        "inputs": sorted(entry["path"] for entry in reports.values()),
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()

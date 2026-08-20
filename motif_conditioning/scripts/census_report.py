"""Score the motif-census arms against the preregistered criteria.

Contrasts are paired over the 1024 fixed validation compositions, because every arm shares
the baseline's warm start, budget and sampler randomness, so a per-composition difference
is far more sensitive than a difference of two aggregate rates.

Example:

    python -m motif_conditioning.scripts.census_report --run-root motif_conditioning/runs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.stats import binomtest, spearmanr


RUN_PATTERN = re.compile(r"^(?P<arm>[A-Z]+)_k(?P<prototypes>\d+)_seed(?P<seed>\d+)$")

SCREEN_POINTS = 1.5
"""Preregistered match-rate gain required of the census against the baseline."""

CONTENT_POINTS = 1.5
"""Preregistered match-rate gain required of the real census against a mismatched one."""

ERROR_TOLERANCE = 0.05
"""Preregistered ceiling on worsening the position or cell endpoint error."""

NOISE_FLOOR_POINTS = 0.68
"""Measured spread across four runs of an identical inference policy."""


def load_outcomes(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    report["matches"] = np.array([entry["match"] for entry in report["outcomes"]], dtype=bool)
    return report


def collect(run_root: Path) -> dict[str, dict[str, Any]]:
    """Index every scored evaluation by arm, vocabulary size, and guidance scale."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(run_root.glob("*/eval/w*/OUTCOMES.json")):
        run = RUN_PATTERN.match(path.parents[2].name)
        if run is None:
            continue
        scale = float(path.parents[0].name[1:])
        report = load_outcomes(path)
        report.update(
            {
                "arm_name": run["arm"],
                "prototypes": int(run["prototypes"]),
                "guidance_scale": scale,
                "path": str(path),
            }
        )
        found[f"{run['arm']}@K={run['prototypes']}@w={scale:g}"] = report
    return found


def contrast(treatment: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Paired match-rate difference with an exact McNemar test on discordant pairs."""
    first, second = treatment["matches"], control["matches"]
    if len(first) != len(second):
        raise ValueError(
            f"{treatment['arm']} scored {len(first)} compositions but "
            f"{control['arm']} scored {len(second)}; they are not paired."
        )
    won = int((first & ~second).sum())
    lost = int((second & ~first).sum())
    return {
        "difference_points": 100.0 * float(first.mean() - second.mean()),
        "won": won,
        "lost": lost,
        "mcnemar_p": float(binomtest(won, won + lost, 0.5).pvalue) if won + lost else 1.0,
        "exceeds_noise_floor": abs(100.0 * float(first.mean() - second.mean()))
        > NOISE_FLOOR_POINTS,
    }


def error_regression(treatment: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Relative change in the endpoint errors the preregistration caps."""
    return {
        field: (treatment[field] - control[field]) / control[field]
        for field in ("position_target_mse", "cell_target_mse")
    }


def information_axis(precheck: Optional[Path]) -> dict[int, float]:
    if precheck is None or not precheck.is_file():
        return {}
    records = json.loads(precheck.read_text())["records"]
    return {
        int(record["prototypes"]): float(record["census_information_beyond_composition"])
        for record in records
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-root", default="motif_conditioning/runs")
    parser.add_argument(
        "--baseline",
        default="joint_geometry/runs/budget2000/oracle/D_seed0/eval/draw0/OUTCOMES.json",
    )
    parser.add_argument("--precheck", default="motif_conditioning/reports/CENSUS-PRECHECK.json")
    parser.add_argument("--out", default="motif_conditioning/reports/CENSUS-GATE.json")
    arguments = parser.parse_args()

    baseline = load_outcomes(Path(arguments.baseline))
    arms = collect(Path(arguments.run_root))
    if not arms:
        raise SystemExit(f"No scored census arms under {arguments.run_root}.")
    information = information_axis(Path(arguments.precheck))

    rows = []
    for key, report in sorted(arms.items()):
        against_baseline = contrast(report, baseline)
        rows.append(
            {
                "arm": key,
                "arm_name": report["arm_name"],
                "prototypes": report["prototypes"],
                "guidance_scale": report["guidance_scale"],
                "match_rate_percent": 100.0 * report["match_rate"],
                "position_target_mse": report["position_target_mse"],
                "cell_target_mse": report["cell_target_mse"],
                "versus_baseline": against_baseline,
                "error_change_versus_baseline": error_regression(report, baseline),
            }
        )

    def conditional(prototypes: int, arm: str = "M", scale: float = 1.0):
        return arms.get(f"{arm}@K={prototypes}@w={scale:g}")

    verdict: dict[str, Any] = {}

    screen = conditional(32)
    if screen is not None:
        difference = contrast(screen, baseline)
        errors = error_regression(screen, baseline)
        verdict["stage_1_screen"] = {
            "difference_points": difference["difference_points"],
            "mcnemar_p": difference["mcnemar_p"],
            "error_change": errors,
            "passed": (
                difference["difference_points"] >= SCREEN_POINTS
                and all(value <= ERROR_TOLERANCE for value in errors.values())
            ),
            "stop_rule_triggered": difference["difference_points"] < 0.0,
        }

    mismatched = conditional(32, arm="X")
    if screen is not None and mismatched is not None:
        difference = contrast(screen, mismatched)
        verdict["stage_2_content"] = {
            "difference_points": difference["difference_points"],
            "mcnemar_p": difference["mcnemar_p"],
            "passed": difference["difference_points"] >= CONTENT_POINTS,
        }

    dose = [
        (information[report["prototypes"]], 100.0 * report["match_rate"])
        for key, report in arms.items()
        if report["arm_name"] == "M"
        and report["guidance_scale"] == 1.0
        and report["prototypes"] in information
    ]
    if len(dose) >= 3:
        axis, rates = zip(*sorted(dose))
        correlation = spearmanr(axis, rates)
        verdict["stage_3_dose_response"] = {
            "information_beyond_composition": list(axis),
            "match_rate_percent": list(rates),
            "spearman_rho": float(correlation.statistic),
            "passed": float(correlation.statistic) > 0.0,
        }

    guidance = sorted(
        (report["guidance_scale"], 100.0 * report["match_rate"])
        for report in arms.values()
        if report["arm_name"] == "M" and report["prototypes"] == 32
    )
    if len(guidance) > 1:
        best = max(guidance, key=lambda item: item[1])
        verdict["stage_4_guidance"] = {
            "curve": [{"scale": scale, "match_rate_percent": rate} for scale, rate in guidance],
            "best_scale": best[0],
            "best_match_rate_percent": best[1],
        }

    report = {
        "preregistration": "motif_conditioning/reports/PREREGISTRATION.json",
        "baseline": {
            "path": arguments.baseline,
            "arm": baseline["arm"],
            "match_rate_percent": 100.0 * baseline["match_rate"],
            "position_target_mse": baseline["position_target_mse"],
            "cell_target_mse": baseline["cell_target_mse"],
        },
        "noise_floor_points": NOISE_FLOOR_POINTS,
        "arms": rows,
        "verdict": verdict,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")

    print(f"baseline {baseline['arm']:24s} {100.0 * baseline['match_rate']:6.2f}%")
    for row in rows:
        print(
            f"{row['arm']:24s} {row['match_rate_percent']:6.2f}%"
            f"  vs D {row['versus_baseline']['difference_points']:+6.2f} pts"
            f"  (won {row['versus_baseline']['won']:3d} lost {row['versus_baseline']['lost']:3d}"
            f"  p={row['versus_baseline']['mcnemar_p']:.4f})"
        )
    for stage, payload in verdict.items():
        if "passed" in payload:
            print(f"{stage}: {'PASS' if payload['passed'] else 'FAIL'}")
    print(f"Wrote {destination}.")


if __name__ == "__main__":
    main()

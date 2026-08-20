"""Read the frozen geometry-loss coefficient from a calibration checkpoint."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from joint_geometry.lightning import CALIBRATION_BATCHES, TARGET_GRADIENT_RATIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="joint_geometry/reports/GEOMETRY-WEIGHT.json")
    arguments = parser.parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    calibration = checkpoint.get("joint_geometry_calibration")
    if not calibration or not calibration.get("calibrated"):
        raise ValueError("The checkpoint does not contain a completed geometry-loss calibration.")
    weight = calibration.get("weight")
    if weight is None or not float(weight) > 0.0:
        raise ValueError(f"The calibrated geometry weight is invalid: {weight}.")
    ratios = [float(value) for value in calibration.get("ratios", [])]
    if len(ratios) != CALIBRATION_BATCHES or not all(value > 0.0 for value in ratios):
        raise ValueError(
            f"Expected {CALIBRATION_BATCHES} positive calibration ratios, "
            f"received {len(ratios)}."
        )
    achieved_ratio = statistics.fmean(float(weight) / value for value in ratios)
    if not math.isclose(
        achieved_ratio,
        TARGET_GRADIENT_RATIO,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "The frozen geometry weight does not achieve the requested mean "
            f"gradient share: {achieved_ratio:.6g} versus {TARGET_GRADIENT_RATIO:.6g}."
        )
    checkpoint_path = Path(arguments.checkpoint)
    report = {
        "checkpoint": arguments.checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "geometry_weight": float(weight),
        "target_shared_trunk_gradient_ratio": TARGET_GRADIENT_RATIO,
        "estimated_shared_trunk_gradient_ratio": achieved_ratio,
        "calibration_batches": len(ratios),
        "raw_flow_to_geometry_gradient_ratios": ratios,
    }
    destination = Path(arguments.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(float(weight))


if __name__ == "__main__":
    main()

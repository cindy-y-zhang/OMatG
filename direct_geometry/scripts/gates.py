"""
Read the gate stamps and say whether they authorise a launch, in one place.

WHY THIS IS NOT A `passed` LOOKUP

The three stamps do not mean the same thing, and reading them as though they did gets the answer wrong:

- **DG0**, the audit, is a bar. Its channels must be finite and varying across the whole probability path, the periodic
  graph must actually carry image multiplicity, and every arm must sit inside the cost ceiling. A failure here means the
  features are not measurable, and nothing downstream can be interpreted.
- **DG1** is a bar. The descriptor must carry information about the local environment that a chemistry-only control does
  not. A failure here ends the pivot, because there would be nothing to inject.
- **DG2** is a *selection*, not a bar. It asks whether the angular channels earn their place on top of the radial ones.
  ``passed: false`` means they did not, so ``radial`` is promoted alone -- which is a finished decision and a perfectly
  good basis for a sweep. Treating it as a failure would refuse to launch precisely because a factor was correctly
  dropped, which is backwards.
- **DG3**, the local screen, is a bar on spending. It authorises the A100 sweep and supports no claim of its own.

So what authorises a launch is: DG0 passed, DG1 passed, some descriptor was promoted, and DG3 passed. DG2's own verdict is
reported as information -- which descriptor, and why -- and never as a veto.

Both launchers and the packager call this rather than each parsing the JSON, so there is one definition of "authorised" and
one definition of "promoted" instead of three that can drift apart.

Usage:

    python -m direct_geometry.scripts.gates                              # print the status of each stamp
    python -m direct_geometry.scripts.gates --require dg0,probes,dg3     # and exit nonzero unless those are satisfied
    python -m direct_geometry.scripts.gates --promoted                   # print only the promoted descriptor
"""

import argparse
import json
from pathlib import Path
from typing import Optional


STAMPS = {"dg0": "DG0-AUDIT.json", "probes": "DG1-DG2-PROBES.json", "dg3": "LOCAL-GATE.json"}
"""Which file each gate's verdict lives in."""


def read_audit(path: Path) -> dict:
    """
    Read the DG0 stamp.

    :param path:
        Path of the stamp.
    :type path: Path

    :return:
        Whether it passed, its failures, and a line to print.
    :rtype: dict
    """
    report = json.loads(path.read_text())
    passed = bool(report.get("passed"))
    failures = report.get("failures", [])
    worst = max((value for entry in report.get("costs", {}).values()
                 for value in (entry.get("time_ratio"), entry.get("memory_ratio"))
                 if value is not None and value == value), default=None)
    summary = "channels finite and varying, arms inside the cost ceiling" if passed else "; ".join(failures[:1])
    if worst is not None:
        summary += f" (worst cost ratio {worst:.2f})"
    return {"passed": passed, "failures": failures, "summary": summary}


def read_probes(path: Path) -> dict:
    """
    Read the DG1/DG2 stamp, treating DG1 as a bar and DG2 as a selection.

    :param path:
        Path of the stamp.
    :type path: Path

    :return:
        Whether it authorises a launch, the promoted descriptor, and a line to print.
    :rtype: dict
    """
    report = json.loads(path.read_text())
    verdict = report.get("verdict", {})
    dg1, dg2 = verdict.get("DG1", {}), verdict.get("DG2", {})
    promoted = verdict.get("promoted")
    failures = []
    if not dg1.get("passed"):
        failures.append("DG1 failed: the descriptor carries no information a chemistry-only control does not, so there "
                        "is nothing to inject and the pivot stops here")
    if promoted is None:
        failures.append("no descriptor was promoted, so arms C and D are undefined")
    angular = "kept" if dg2.get("passed") else "dropped"
    summary = (f"DG1 +{dg1.get('accuracy_points', float('nan')):.1f} points and "
               f"{dg1.get('bits', float('nan')):.2f} bits over the chemistry-only control; angular channels {angular}; "
               f"promoted {promoted!r}")
    return {"passed": not failures, "failures": failures, "promoted": promoted, "angular_kept": bool(dg2.get("passed")),
            "summary": summary}


def read_local_gate(path: Path) -> dict:
    """
    Read the DG3 stamp.

    :param path:
        Path of the stamp.
    :type path: Path

    :return:
        Whether it passed, its failures, and a line to print.
    :rtype: dict
    """
    report = json.loads(path.read_text())
    verdict = report.get("verdict", {})
    passed = bool(verdict.get("passed"))
    failures = verdict.get("failures", [])
    paired = report.get("contrasts", {}).get("paired", {}).get("match", {}).get("mean_points")
    summary = "the local screen authorises the sweep" if passed else "; ".join(failures[:1])
    if paired is not None:
        summary += f" (paired D-A {paired:+.2f} points)"
    return {"passed": passed, "failures": failures, "summary": summary}


READERS = {"dg0": read_audit, "probes": read_probes, "dg3": read_local_gate}
"""How to read each stamp. Separate functions because the three do not mean the same thing."""


def authorisation(reports: Path, require: tuple[str, ...]) -> dict:
    """
    Read every stamp and decide whether the required ones authorise a launch.

    :param reports:
        Directory holding the stamps.
    :type reports: Path
    :param require:
        Gates that must be satisfied. An unsatisfied gate outside this set is reported and not enforced, which is what
        makes an informational status read possible without duplicating the reading.
    :type require: tuple[str, ...]

    :return:
        Per-gate status, the promoted descriptor, whether the requirement is met, and why not.
    :rtype: dict
    """
    status: dict[str, dict] = {}
    problems: list[str] = []
    for gate, filename in STAMPS.items():
        path = reports / filename
        if not path.is_file():
            status[gate] = {"present": False, "passed": False, "summary": f"{filename} is missing from {reports}"}
            if gate in require:
                problems.append(status[gate]["summary"])
            continue
        try:
            entry = READERS[gate](path)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            status[gate] = {"present": True, "passed": False,
                            "summary": f"{filename} could not be read: {error}"}
            if gate in require:
                problems.append(status[gate]["summary"])
            continue
        status[gate] = {"present": True, **entry}
        if gate in require and not entry["passed"]:
            problems.extend(entry["failures"] or [f"{filename} did not pass"])

    return {"status": status, "promoted": status.get("probes", {}).get("promoted"),
            "authorised": not problems, "problems": problems, "required": list(require)}


def main(argv: Optional[list[str]] = None) -> int:
    """
    Print the gate status, or only the promoted descriptor.

    :param argv:
        Command line arguments, or None to read them from the process.
    :type argv: Optional[list[str]]

    :return:
        Zero if every required gate is satisfied, one otherwise.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", default="direct_geometry/reports", help="directory holding the stamps")
    parser.add_argument("--require", default="", help="comma-separated gates that must pass: dg0, probes, dg3")
    parser.add_argument("--promoted", action="store_true",
                        help="print only the promoted descriptor, for a launcher to read")
    settings = parser.parse_args(argv)

    require = tuple(part.strip() for part in settings.require.split(",") if part.strip())
    unknown = [gate for gate in require if gate not in STAMPS]
    if unknown:
        parser.error(f"unknown gate(s) {unknown}; choose from {sorted(STAMPS)}")

    report = authorisation(Path(settings.reports), require)

    if settings.promoted:
        # Requiring the probes here rather than trusting the caller: a launcher that read an absent descriptor would
        # train arms C and D on whatever the config happened to default to.
        probes = report["status"].get("probes", {})
        if not probes.get("passed"):
            print(f"the descriptor is undefined: {probes.get('summary', 'no probe report')}\n"
                  f"Run 'python -m direct_geometry.scripts.probe_features' first. Gates DG1 and DG2 choose it, and a "
                  f"run that picked its own would be answering a question nothing asked.", flush=True)
            return 1
        print(report["promoted"])
        return 0

    for gate, filename in STAMPS.items():
        entry = report["status"][gate]
        mark = "pass" if entry["passed"] else ("FAIL" if entry["present"] else "missing")
        needed = " (required)" if gate in require else ""
        print(f"  {filename:<22} {mark:<8}{needed}")
        print(f"    {entry['summary']}")

    if report["problems"]:
        print("\nnot authorised:")
        for problem in report["problems"]:
            print(f"  - {problem}")
        return 1
    if require:
        print(f"\nauthorised by {', '.join(require)}; promoted descriptor {report['promoted']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

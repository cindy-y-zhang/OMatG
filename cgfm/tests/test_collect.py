"""
Tests for collecting results across runs.

The Stage 1 eta sweep is decided entirely on the best validation match rate read back out of the CSV logs, so the
reading has to be right about the things that actually happen in those files: match rate is logged only on validation
epochs, which leaves the cell blank on every other row, and a run can be resumed into more than one version directory.
"""

import json
from pathlib import Path
from cgfm.scripts.collect_results import best_validation_match_rate, collect_test, read_score


def _write_metrics(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    """
    Write a Lightning-style metrics.csv.

    :param path:
        Destination path.
    :type path: Path
    :param rows:
        Rows to write, as mappings from column name to value.
    :type rows: list[dict[str, str]]
    :param columns:
        Column names, in order.
    :type columns: list[str]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    lines.extend(",".join(row.get(column, "") for column in columns) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def test_best_validation_match_rate_skips_blank_rows(tmp_path):
    """Lightning leaves the match rate blank on training-only rows, which must not be read as a value."""
    _write_metrics(tmp_path / "version_0" / "metrics.csv",
                   [{"epoch": "0", "loss_total_epoch": "2.5"},
                    {"epoch": "0", "match_rate": "0.12"},
                    {"epoch": "1", "loss_total_epoch": "2.1"},
                    {"epoch": "1", "match_rate": "0.31"}],
                   ["epoch", "loss_total_epoch", "match_rate"])
    assert best_validation_match_rate(tmp_path) == 31.0


def test_best_validation_match_rate_spans_resumed_runs(tmp_path):
    """A resumed run writes a second version directory, and the best over the whole run is what matters."""
    _write_metrics(tmp_path / "version_0" / "metrics.csv", [{"match_rate": "0.20"}], ["match_rate"])
    _write_metrics(tmp_path / "version_1" / "metrics.csv", [{"match_rate": "0.45"}], ["match_rate"])
    assert best_validation_match_rate(tmp_path) == 45.0


def test_best_validation_match_rate_is_none_without_a_match_rate(tmp_path):
    """A run validated on loss logs no match rate, which must read as absent rather than as zero."""
    _write_metrics(tmp_path / "version_0" / "metrics.csv", [{"val_loss_total": "2.5"}], ["val_loss_total"])
    assert best_validation_match_rate(tmp_path) is None


def test_missing_scores_are_reported_not_dropped(tmp_path):
    """A partial table must say which runs are missing, or it could be mistaken for a complete one."""
    score = {"one_shot": {"match_rate": 0.4, "mean_crmse": 0.2}, "one_shot_metre": {"match_rate": 0.5},
             "best_of_n": {"match_rate": 0.6, "mean_crmse": 0.1}}
    path = tmp_path / "shells" / "seed0" / "eval" / "nfe210" / "score.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(score))
    (tmp_path / "atomwise" / "seed0").mkdir(parents=True)

    collected, missing = collect_test(tmp_path, nfe=210)
    assert collected["shells"]["one-shot match"] == [40.0]
    assert collected["atomwise"] == {}
    assert any("atomwise/seed0" in entry for entry in missing)


def test_read_score_converts_rates_to_percentages(tmp_path):
    """Rates are stored as fractions and reported as percentages, so the conversion must happen exactly once."""
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"one_shot": {"match_rate": 0.125, "mean_crmse": 0.25}}))
    values = read_score(path)
    assert values["one-shot match"] == 12.5
    assert values["one-shot cRMSE"] == 0.25

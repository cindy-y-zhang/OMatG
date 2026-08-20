"""Configuration-level checks for the seven fixed experimental arms."""

from pathlib import Path

import yaml


CONFIGS = Path("joint_geometry/configs")


def _config(arm: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{arm}.yaml").read_text())


def test_all_preregistered_arm_configs_exist() -> None:
    assert {path.stem for path in CONFIGS.glob("[SDRHOPJ].yaml")} == {
        "S",
        "D",
        "R",
        "H",
        "O",
        "P",
        "J",
    }


def test_direct_controls_select_only_the_intended_current_state_input() -> None:
    direct = _config("D")["model"]["model"]["init_args"]["encoder"]["init_args"]
    recomputed = _config("R")["model"]["model"]["init_args"]["encoder"]["init_args"]
    assert direct["feature_mode"] == "none"
    assert recomputed["feature_mode"] == "radial"
    assert direct["message_graph"] == recomputed["message_graph"] == "fc"


def test_joint_controls_differ_only_in_declared_semantics() -> None:
    head = _config("H")
    shuffled = _config("P")
    real = _config("J")
    assert not head["model"]["model"]["init_args"]["encoder"]["init_args"][
        "geometry_input"
    ]
    assert shuffled["model"]["model"]["init_args"]["encoder"]["init_args"][
        "geometry_input"
    ]
    assert real["model"]["model"]["init_args"]["encoder"]["init_args"][
        "geometry_input"
    ]
    assert not head["data"]["shuffle_within_element"]
    assert shuffled["data"]["shuffle_within_element"]
    assert not real["data"]["shuffle_within_element"]


def test_oracle_leakage_is_confined_to_o_arm() -> None:
    oracle = _config("O")
    assert (
        oracle["model"]["sampler"]["class_path"]
        == "joint_geometry.sampler.OracleGeometrySampler"
    )
    assert (
        oracle["model"]["si"]["class_path"]
        == "joint_geometry.interpolants.OracleGeometryInterpolants"
    )
    for arm in ("H", "P", "J"):
        text = (CONFIGS / f"{arm}.yaml").read_text()
        assert "OracleGeometry" not in text


def test_default_joint_config_uses_the_promoted_descriptor_and_backbone() -> None:
    base = yaml.safe_load((CONFIGS / "mpts52.yaml").read_text())
    assert base["data"]["geometry_dir"].endswith("/radial17")
    assert base["model"]["sampler"]["init_args"]["geometry_dimension"] == 17
    encoder = base["model"]["model"]["init_args"]["encoder"]["init_args"]
    assert encoder["geometry_dimension"] == 17
    assert encoder["message_graph"] == "fc"
    assert _config("O")["model"]["sampler"]["init_args"]["geometry_dimension"] == 17

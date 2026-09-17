"""Customer-facing Policy Space CLI tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from policy_space.cli import main


def test_init_creates_four_file_policy_template(tmp_path: Path) -> None:
    result = main(
        [
            "init",
            "customer_policy",
            "--root",
            str(tmp_path),
            "--observation",
            "single_arm_rgb_joint@1",
            "--action",
            "single_arm_joint_position@1",
        ]
    )

    policy_dir = tmp_path / "customer_policy"
    assert result == 0
    assert sorted(path.name for path in policy_dir.iterdir()) == [
        "README.md",
        "deploy.yaml",
        "model.py",
        "requirements.txt",
    ]
    assert "single_arm_rgb_joint@1" in (policy_dir / "deploy.yaml").read_text(
        encoding="utf-8"
    )
    deploy = yaml.safe_load((policy_dir / "deploy.yaml").read_text(encoding="utf-8"))
    assert deploy["server"]["host"] == "127.0.0.1"
    assert deploy["server"]["max_payload_depth"] == 32
    assert deploy["model"]["cfg"]["action_dim"] == 8


def test_init_can_group_policy_under_a_family(tmp_path: Path) -> None:
    result = main(
        [
            "init",
            "customer_policy",
            "--root",
            str(tmp_path),
            "--family",
            "customer_family",
            "--observation",
            "single_arm_rgb_joint@1",
            "--action",
            "single_arm_joint_position@1",
        ]
    )

    policy_dir = tmp_path / "customer_family" / "customer_policy"
    assert result == 0
    assert (policy_dir / "deploy.yaml").is_file()
    deploy = yaml.safe_load((policy_dir / "deploy.yaml").read_text(encoding="utf-8"))
    assert deploy["policy_name"] == "customer_policy"


def test_init_rejects_unknown_contracts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown observation schema"):
        main(
            [
                "init",
                "customer_policy",
                "--root",
                str(tmp_path),
                "--observation",
                "does_not_exist@9",
                "--action",
                "single_arm_joint_position@1",
            ]
        )


def test_check_runs_complete_episode_for_example_policy(capsys) -> None:
    deploy = (
        Path(__file__).resolve().parents[1]
        / "policies"
        / "example_policy"
        / "deploy.yaml"
    )

    assert main(["check", str(deploy)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["protocol_version"] == "2"
    assert report["schema_pair"] == {
        "observation": "bimanual_rgb_ee@1",
        "action": "bimanual_ee_absolute@1",
        "embodiment": "ex001_6r@1",
    }
    assert report["action"]["shape"] == [32, 14]
    assert report["lifecycle"] == [
        "initialize_episode",
        "ingest_observation",
        "infer_actions",
        "finalize_episode",
    ]


def test_check_requires_selection_for_multiple_schema_pairs(
    tmp_path: Path, capsys
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "policies" / "templates" / "bimanual_ee"
    )
    policy_dir = tmp_path / "multi_policy"
    policy_dir.mkdir()
    shutil.copy2(source / "model.py", policy_dir / "model.py")
    deploy = yaml.safe_load((source / "deploy.yaml").read_text(encoding="utf-8"))
    deploy["metadata"]["supported_schema_pairs"].append(
        {
            "observation_schema": "bimanual_rgb_joint@1",
            "action_schema": "bimanual_ee_absolute@1",
            "embodiment_constraints": ["ex001_6r@1"],
        }
    )
    deploy_path = policy_dir / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple schema pairs"):
        main(["check", str(deploy_path)])

    assert (
        main(
            [
                "check",
                str(deploy_path),
                "--observation",
                "bimanual_rgb_ee@1",
                "--action",
                "bimanual_ee_absolute@1",
                "--embodiment",
                "ex001_6r@1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

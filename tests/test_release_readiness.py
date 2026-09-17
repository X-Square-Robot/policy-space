"""Release-readiness guards for the standalone Policy Space repository."""

from __future__ import annotations

import ipaddress
import re
import tomllib
from pathlib import Path

import yaml

import policy_space
from policy_space.runtime.server import load_policy_definition

ROOT = Path(__file__).resolve().parents[1]
TEXT_FILE_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
MACHINE_DATA_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_.-])/(?:home|Users)/[A-Za-z0-9._-]+"),
    re.compile(r"(?<![A-Za-z0-9_.-])/mnt/[A-Za-z0-9._-]+"),
    re.compile(
        r"(?im)(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9-]+\.)+local(?=$|[:/\s])"
    ),
)
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def _contains_sensitive_machine_data(text: str) -> bool:
    if any(pattern.search(text) for pattern in MACHINE_DATA_PATTERNS):
        return True

    for candidate in IPV4_PATTERN.findall(text):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            return True
    return False


def _find_sensitive_text_offenders(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _contains_sensitive_machine_data(text):
            offenders.append(str(path.relative_to(root)))
    return sorted(offenders)


def test_sensitive_text_scanner_detects_generic_machine_identifiers(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text(
        "/" + "home/example-user/project\n"
        "/" + "mnt/private-volume/checkpoint\n"
        "build.example." + "local\n"
        "198.51.100." + "42\n",
        encoding="utf-8",
    )

    assert _find_sensitive_text_offenders(tmp_path) == ["sample.md"]


PRODUCTION_POLICIES = {
    "cosmos3_droid": "cosmos3/cosmos3_droid_joint",
    "cosmos3_franka": "cosmos3/cosmos3_franka_joint",
    "cosmos3_x2robot": "cosmos3/cosmos3_artixon_arm_6a_eef",
    "dm05_x2real": "dm05/dm05_artixon_arm_6a_joint",
    "dm05_x2real_ee": "dm05/dm05_artixon_arm_6a_eef",
    "dreamzero_droid": "dreamzero/dreamzero_droid_joint",
    "dreamzero_x2robot": "dreamzero/dreamzero_artixon_arm_6a_joint",
    "fastwam_x2robot": "fastwam/fastwam_artixon_arm_6a_joint",
    "openpi_arx_ee": "openpi/pi05_arx_r5_eef",
    "openpi_arx_joint": "openpi/pi05_arx_r5_joint",
    "openpi_x2real_joint": "openpi/pi05_artixon_arm_6a_joint",
    "openpi_x2robot": "openpi/pi05_artixon_arm_6a_eef",
    "pi05_ex001_wholebody": "openpi/pi05_quanta_x1_whole_body",
    "smolvla_franka": "smolvla/smolvla_franka_joint",
    "wallx": "wallx/wall_oss_05_artixon_arm_6a_eef",
    "wallx_ex001_wholebody": "wallx/wallx_quanta_x1_whole_body",
}

PRODUCTION_POLICY_TITLES = {
    "cosmos3/cosmos3_artixon_arm_6a_eef": "Cosmos3 ArtiXon Arm-6A EEF",
    "cosmos3/cosmos3_droid_joint": "Cosmos3 DROID Joint",
    "cosmos3/cosmos3_franka_joint": "Cosmos3 Franka Joint",
    "dm05/dm05_artixon_arm_6a_eef": "DM0.5 ArtiXon Arm-6A EEF",
    "dm05/dm05_artixon_arm_6a_joint": "DM0.5 ArtiXon Arm-6A Joint",
    "dreamzero/dreamzero_artixon_arm_6a_joint": "DreamZero ArtiXon Arm-6A Joint",
    "dreamzero/dreamzero_droid_joint": "DreamZero DROID Joint",
    "fastwam/fastwam_artixon_arm_6a_joint": "FastWAM ArtiXon Arm-6A Joint",
    "openpi/pi05_artixon_arm_6a_eef": "OpenPI (π0.5) ArtiXon Arm-6A EEF",
    "openpi/pi05_artixon_arm_6a_joint": "OpenPI (π0.5) ArtiXon Arm-6A Joint",
    "openpi/pi05_arx_r5_eef": "OpenPI (π0.5) ARX R5 EEF",
    "openpi/pi05_arx_r5_joint": "OpenPI (π0.5) ARX R5 Joint",
    "openpi/pi05_quanta_x1_whole_body": "OpenPI (π0.5) Quanta X1 Whole Body",
    "smolvla/smolvla_franka_joint": "SmolVLA Franka Joint",
    "wallx/wall_oss_05_artixon_arm_6a_eef": "Wall-X (wall-oss-0.5) ArtiXon Arm-6A EEF",
    "wallx/wallx_quanta_x1_whole_body": "Wall-X Quanta X1 Whole Body",
}


def test_production_policy_catalog_is_complete() -> None:
    missing: list[str] = []
    for name, relative_path in sorted(PRODUCTION_POLICIES.items()):
        policy_dir = ROOT / "policies" / relative_path
        for filename in ("README.md", "deploy.yaml", "model.py", "requirements.txt"):
            if not (policy_dir / filename).is_file():
                missing.append(f"{name}/{filename}")
    assert missing == []


def test_production_readme_titles_match_integration_names() -> None:
    assert set(PRODUCTION_POLICY_TITLES) == set(PRODUCTION_POLICIES.values())
    for relative_path, title in PRODUCTION_POLICY_TITLES.items():
        policy_dir = ROOT / "policies" / relative_path
        for filename in ("README.md", "README.zh-CN.md"):
            first_line = (policy_dir / filename).read_text(encoding="utf-8").splitlines()[0]
            assert first_line == f"# {title}"


def test_production_deployments_resolve_from_catalog_paths() -> None:
    for relative_path in PRODUCTION_POLICIES.values():
        policy_dir = ROOT / "policies" / relative_path
        definition = load_policy_definition(policy_dir)
        assert definition.model_path.parent == policy_dir


def test_readme_integration_names_match_linked_directories() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\|[^\n]+\|[^\n]+\| \[`(?P<name>[^`]+)`\]"
        r"\((?P<path>policies/[^)]+)\) \|$",
        readme,
        flags=re.MULTILINE,
    )

    assert len(rows) == len(PRODUCTION_POLICIES)
    for name, relative_path in rows:
        assert Path(relative_path).name == name
        assert (ROOT / relative_path).is_dir()


def test_gitlab_ci_builds_and_tests_the_distribution() -> None:
    ci_path = ROOT / ".gitlab-ci.yml"
    assert ci_path.is_file()
    ci = ci_path.read_text(encoding="utf-8")
    assert "pytest" in ci
    assert "python -m build" in ci
    assert "dist/" in ci
    assert "packages/protocol" in ci
    assert "policy_space_protocol" in ci


def test_release_documentation_covers_immutable_installation() -> None:
    release_path = ROOT / "docs" / "release.md"
    assert release_path.is_file()
    release = release_path.read_text(encoding="utf-8")
    assert "v0.3.0" in release
    assert "<repository-url>" in release
    assert not _contains_sensitive_machine_data(release)
    assert "dist/policy_space-" in release
    assert "git archive" in release
    assert "git init" in release


def test_public_tree_excludes_internal_working_notes_and_machine_paths() -> None:
    assert not (ROOT / "docs" / "superpowers").exists()
    assert _find_sensitive_text_offenders(ROOT) == []


def test_policy_guides_use_the_public_cli() -> None:
    offenders: list[str] = []
    for path in (ROOT / "policies").rglob("README.md"):
        if "python -m policy_space.server" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_model_contract_examples_do_not_reuse_the_wire_protocol_field() -> None:
    examples = sorted((ROOT / "policies").rglob("x2robot_contract.example.json"))
    assert examples
    for path in examples:
        document = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert document["model_contract_version"] == 1
        assert "protocol_version" not in document


def test_package_version_is_release_tag_compatible() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = document["project"]["version"]
    protocol_document = tomllib.loads(
        (ROOT / "packages" / "protocol" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert version == "0.3.0"
    assert protocol_document["project"]["version"] == version
    assert policy_space.__version__ == version


def test_normalized_gripper_policies_declare_their_wire_convention() -> None:
    for policy_name in ("wallx", "dreamzero_x2robot"):
        deploy = yaml.safe_load(
            (
                ROOT / "policies" / PRODUCTION_POLICIES[policy_name] / "deploy.yaml"
            ).read_text(encoding="utf-8")
        )
        assert deploy["metadata"]["gripper_convention"] == "normalized_01"

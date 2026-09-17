"""Standalone repository packaging and dependency-boundary tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from policy_space.contracts.registry import ContractRegistry

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "policy_space"
TEMPLATE_PATHS = {
    "example_policy": ROOT / "policies" / "templates" / "bimanual_ee",
    "example_franka_policy": ROOT / "policies" / "templates" / "franka_joint",
    "example_mobile_policy": ROOT / "policies" / "templates" / "mobile_manipulation",
}
POLICY_FAMILIES = {
    "cosmos3",
    "dm05",
    "dreamzero",
    "fastwam",
    "openpi",
    "wallx",
}
USER_DOCS = {
    "README.md",
    "policy-integration.md",
    "manaenv-evaluation.md",
    "contracts.md",
    "deployment.md",
    "troubleshooting.md",
}


def test_standalone_package_uses_src_layout() -> None:
    assert (ROOT / "pyproject.toml").is_file()
    assert (PACKAGE_ROOT / "__init__.py").is_file()
    assert (PACKAGE_ROOT / "runtime" / "server.py").is_file()
    assert (PACKAGE_ROOT / "adapters" / "base.py").is_file()


def test_user_documentation_has_a_complete_navigation_layer() -> None:
    docs_root = ROOT / "docs"
    assert docs_root.is_dir()
    assert USER_DOCS <= {path.name for path in docs_root.glob("*.md")}
    for filename in USER_DOCS:
        if filename == "README.md":
            chinese_name = "README.zh-CN.md"
        else:
            chinese_name = filename.removesuffix(".md") + ".zh-CN.md"
        assert (docs_root / chinese_name).is_file()


def test_public_markdown_local_links_resolve() -> None:
    broken: list[str] = []
    markdown_paths = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
    markdown_paths.extend((ROOT / "docs").rglob("*.md"))
    markdown_paths.extend((ROOT / "policies").rglob("*.md"))

    for path in markdown_paths:
        content = path.read_text(encoding="utf-8")
        for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if destination.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = destination.split("#", 1)[0].split("?", 1)[0]
            if not (path.parent / relative_target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {destination}")
    assert broken == []


def test_public_markdown_links_keep_the_selected_language() -> None:
    mixed: list[str] = []
    markdown_paths = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
    markdown_paths.extend((ROOT / "docs").rglob("*.md"))
    markdown_paths.extend((ROOT / "policies").rglob("*.md"))

    for path in markdown_paths:
        is_chinese = path.name.endswith(".zh-CN.md")
        content = path.read_text(encoding="utf-8")
        for label, destination in re.findall(
            r"\[([^\]]*)\]\(([^)]+\.md(?:#[^)]*)?)\)", content
        ):
            destination_is_chinese = ".zh-CN.md" in destination
            is_language_switch = label in {"English", "简体中文"}
            if destination_is_chinese != is_chinese and not is_language_switch:
                mixed.append(f"{path.relative_to(ROOT)} -> {destination}")
    assert mixed == []


def test_repository_root_exposes_only_primary_user_and_package_boundaries() -> None:
    assert not (ROOT / "examples").exists()
    assert not (ROOT / "contracts").exists()
    assert (ROOT / "docs").is_dir()
    assert (ROOT / "policies").is_dir()
    assert (ROOT / "src").is_dir()
    assert (ROOT / "tests").is_dir()


def test_policy_catalog_contains_documented_templates() -> None:
    assert (ROOT / "policies" / "README.md").is_file()
    assert (ROOT / "policies" / "README.zh-CN.md").is_file()
    assert (ROOT / "policies" / "templates" / "README.md").is_file()
    assert (ROOT / "policies" / "templates" / "README.zh-CN.md").is_file()
    for path in TEMPLATE_PATHS.values():
        assert (path / "README.md").is_file()
        assert (path / "README.zh-CN.md").is_file()
        assert (path / "deploy.yaml").is_file()
        assert (path / "model.py").is_file()
        assert (path / "requirements.txt").is_file()


def _policy_deployments() -> list[Path]:
    return sorted((ROOT / "policies").rglob("deploy.yaml"))


def test_every_policy_and_template_has_a_readme() -> None:
    missing: list[str] = []
    for deploy_yaml in _policy_deployments():
        for filename in ("README.md", "README.zh-CN.md"):
            if not (deploy_yaml.parent / filename).is_file():
                missing.append(f"{deploy_yaml.parent.relative_to(ROOT)}/{filename}")
    assert missing == []


def test_production_policies_are_grouped_by_family() -> None:
    policies_root = ROOT / "policies"
    for family in POLICY_FAMILIES:
        assert (policies_root / family / "README.md").is_file()
        assert (policies_root / family / "README.zh-CN.md").is_file()
    assert not list(policies_root.glob("*/deploy.yaml"))


def test_runtime_package_does_not_import_manaenv() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "manaenv" or name.startswith("manaenv.") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_shipped_policies_declare_session_mode_explicitly() -> None:
    missing: list[str] = []
    for path in _policy_deployments():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not (document.get("server") or {}).get("session_mode"):
            missing.append(str(path.relative_to(ROOT)))
    assert missing == []


def test_shipped_policy_schema_pairs_resolve_in_public_registry() -> None:
    registry = ContractRegistry.builtin()
    missing: list[str] = []
    invalid: list[str] = []
    for path in _policy_deployments():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pairs = (document.get("metadata") or {}).get("supported_schema_pairs") or []
        if not pairs:
            missing.append(str(path.relative_to(ROOT)))
            continue
        for pair in pairs:
            if pair.get("observation_schema") not in registry.observations:
                invalid.append(f"{path.name}:{pair.get('observation_schema')}")
            if pair.get("action_schema") not in registry.actions:
                invalid.append(f"{path.name}:{pair.get('action_schema')}")
            for embodiment in pair.get("embodiment_constraints", []):
                if embodiment not in registry.embodiments:
                    invalid.append(f"{path.name}:{embodiment}")
    assert missing == []
    assert invalid == []


def test_public_repository_does_not_name_external_project() -> None:
    offenders: list[str] = []
    public_roots = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs",
        PACKAGE_ROOT / "contracts",
        ROOT / "policies",
    ]
    external_project_name = "xpolicy" + "lab"
    for root in public_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in {
                ".md",
                ".py",
                ".yaml",
                ".yml",
                ".json",
            }:
                if (
                    external_project_name
                    in path.read_text(encoding="utf-8", errors="ignore").lower()
                ):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

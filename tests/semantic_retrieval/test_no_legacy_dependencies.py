"""Executable architecture boundary, including negative scanner fixtures."""

import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY = {
    "main_task_retrieval",
    "metrics",
    "modules",
    "query_models",
    "prob_models",
    "uncertainty_modules",
    "crf_ladder",
    "dataloaders",
    "util",
}
SELECTORS = ("use_uatvr_full", "registered_contextual_risk_a3", "--oq1_", "--csd_logvar", "--use_crf")


def violations(path):
    source = path.read_text()
    found = []
    if path.suffix == ".py":
        tree = ast.parse(source)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.func
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__import__"
                    or isinstance(target, ast.Attribute)
                    and target.attr == "import_module"
                ):
                    modules = [node.args[0].value.lstrip(".")]
            if any(name.split(".")[0] in LEGACY for name in modules):
                found.append(f"{path}:{node.lineno}: legacy import")
    elif path.suffix == ".sh" and re.search(r"(main_task_retrieval|run_train_|modules/modeling)", source):
        found.append(f"{path}: retired shell entry")
    if any(selector in source for selector in SELECTORS):
        found.append(f"{path}: retired selector")
    return found


def test_old_model_entrypoints_are_removed():
    for name in ("main_task_retrieval.py", "metrics.py", "modules/modeling.py", "modules/modeling_mulit.py"):
        assert not (ROOT / name).exists(), name


def test_active_sources_do_not_import_or_select_legacy_architectures():
    names = subprocess.check_output(["git", "ls-files", "*.py", "*.sh"], cwd=ROOT, text=True).splitlines()
    paths = [ROOT / name for name in names if not name.startswith("tests/") and (ROOT / name).exists()]
    failures = [item for path in paths for item in violations(path)]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "source",
    [
        "from modules.modeling import UATVR",
        "from .prob_models import head",
        'import importlib\nimportlib.import_module("modules.modeling_mulit")',
        '__import__("query_models.head")',
    ],
)
def test_dependency_scan_catches_injected_bad_references(tmp_path, source):
    path = tmp_path / "bad.py"
    path.write_text(source)
    assert violations(path)


def test_dependency_scan_catches_shell_entry(tmp_path):
    path = tmp_path / "bad.sh"
    path.write_text("python main_task_retrieval.py --use_uatvr_full")
    assert violations(path)


@pytest.mark.parametrize(
    "script",
    [
        "manage_experiment_run.py",
        "build_msrvtt_trusted_split.py",
        "inventory_legacy_artifacts.py",
        "migrate_critical_artifacts.py",
    ],
)
def test_supported_cli_help_without_legacy_imports(script):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_new_package_import_closure():
    names = subprocess.check_output(
        ["git", "ls-files", "src/semantic_retrieval/*.py", "src/semantic_retrieval/**/*.py"], cwd=ROOT, text=True
    ).splitlines()
    for name in names:
        module = ".".join(Path(name).relative_to("src").with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[:-9]
        importlib.import_module(module)
    assert not LEGACY.intersection(sys.modules)

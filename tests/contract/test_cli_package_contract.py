from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLI_MAIN = "visual_events_cli.main:main"
SERVER_MAIN = "visual_events_server.app:main"
HEAVY_IMPORTS = {
    "torch",
    "ultralytics",
    "visual_events_server.inference.factory",
    "visual_events_server.inference.ultralytics_pose",
}


def load_pyproject() -> dict:
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)


def test_distribution_name_stays_server_and_adds_cli_entrypoint():
    project = load_pyproject()["project"]
    scripts = project["scripts"]

    assert project["name"] == "visual-events-server"
    assert scripts["visual-events-server"] == SERVER_MAIN
    assert scripts["visual-events-cli"] == CLI_MAIN


def test_importing_cli_main_does_not_import_heavy_inference_stack():
    env = os.environ.copy()
    pythonpath = os.pathsep.join(
        str(path)
        for path in [
            REPO_ROOT / "src",
            REPO_ROOT,
            env.get("PYTHONPATH", ""),
        ]
        if str(path)
    )
    env["PYTHONPATH"] = pythonpath

    code = f"""
import importlib
import sys

importlib.import_module("visual_events_cli.main")
heavy_imports = {sorted(HEAVY_IMPORTS)!r}
loaded = [name for name in heavy_imports if name in sys.modules]
if loaded:
    raise SystemExit("CLI import loaded heavy modules: " + ", ".join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

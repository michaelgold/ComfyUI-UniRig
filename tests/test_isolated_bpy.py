import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "isolated_bpy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("isolated_bpy", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blender_worker_environment_drops_open3d_preload():
    with patch.dict(
        os.environ,
        {"LD_PRELOAD": "/app/.venv/open3d/libOpen3D.so", "KEEP_ME": "yes"},
        clear=True,
    ):
        worker_env = _load_module().blender_worker_environment()

        assert "LD_PRELOAD" not in worker_env
        assert worker_env["KEEP_ME"] == "yes"
        assert os.environ["LD_PRELOAD"] == "/app/.venv/open3d/libOpen3D.so"

    mia_source = (ROOT / "nodes" / "mia_inference.py").read_text()
    assert "env=blender_worker_environment()" in mia_source


if __name__ == "__main__":
    test_blender_worker_environment_drops_open3d_preload()

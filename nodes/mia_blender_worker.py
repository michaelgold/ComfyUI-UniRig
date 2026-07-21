"""Isolated Blender/bpy worker for Make-It-Animatable exports.

This script is launched as a subprocess by mia_inference.py. Keeping bpy imports
inside this process prevents Blender exporter segfaults from terminating the
ComfyUI server process.

Boundary format:
- mesh is passed as a real temporary GLB file
- rig data is passed as an explicit NumPy .npz archive
- no pickle/Python object serialization is used
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comfy_root() -> Path:
    # /app/comfy/custom_nodes/ComfyUI-UniRig/nodes/mia_blender_worker.py
    return Path(__file__).resolve().parents[3]


def _load_mia_export_function():
    repo_root = _repo_root()
    nodes_dir = repo_root / "nodes"
    comfy_root = _comfy_root()
    for path in (comfy_root, repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Avoid executing nodes/__init__.py, which imports ComfyUI node API modules
    # unnecessary for this worker. We only need mia_inference plus its relative
    # imports, so create a synthetic package with the right __path__.
    package = sys.modules.get("nodes")
    if package is None:
        package = types.ModuleType("nodes")
        package.__path__ = [str(nodes_dir)]
        package.__package__ = "nodes"
        sys.modules["nodes"] = package

    module_name = "nodes.mia_inference"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, nodes_dir / "mia_inference.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load nodes.mia_inference for MIA Blender worker")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "nodes"
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module._export_mia_fbx_direct


def _load_npz_payload(payload_path: str) -> dict:
    import numpy as np

    payload = np.load(payload_path, allow_pickle=False)
    bone_names = payload["bone_names"].astype(str).tolist()
    bone_indices = payload["bone_indices"].astype(int).tolist()
    data = {
        "mesh_path": str(payload["mesh_path"].item()),
        "joints": payload["joints"],
        "joints_tail": payload["joints_tail"] if bool(payload["has_joints_tail"].item()) else None,
        "bw": payload["bw"],
        "pose": payload["pose"] if bool(payload["has_pose"].item()) else None,
        "bones_idx_dict": dict(zip(bone_names, bone_indices)),
        "parent_indices": payload["parent_indices"].astype(int).tolist(),
        "pose_ignore_list": [],
    }
    return data


def export_mia(payload_path: str, output_path: str, template_path: str, *, remove_fingers: bool, reset_to_rest: bool, embed_textures: bool) -> None:
    # Import only inside this worker process. This pulls in bpy lazily inside
    # _export_mia_fbx_direct, not in the ComfyUI server.
    export_direct = _load_mia_export_function()

    data = _load_npz_payload(payload_path)

    export_direct(
        data,
        output_path,
        remove_fingers=remove_fingers,
        reset_to_rest=reset_to_rest,
        template_path=Path(template_path),
        embed_textures=embed_textures,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated MIA Blender export worker")
    parser.add_argument("payload_path", help="MIA export payload .npz")
    parser.add_argument("output_path", help="Output FBX/GLB path")
    parser.add_argument("template_path", help="Mixamo template FBX path")
    parser.add_argument("--remove-fingers", action="store_true")
    parser.add_argument("--reset-to-rest", action="store_true")
    parser.add_argument("--embed-textures", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_mia(
        args.payload_path,
        args.output_path,
        args.template_path,
        remove_fingers=args.remove_fingers,
        reset_to_rest=args.reset_to_rest,
        embed_textures=args.embed_textures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Environment isolation for Blender worker subprocesses."""

from __future__ import annotations

import os


def blender_worker_environment() -> dict[str, str]:
    """Copy the process environment without Open3D's global preload.

    ComfyUI preloads Open3D on Linux ARM64 to satisfy its static-TLS needs.
    Blender's bundled OpenVDB/TBB stack is ABI-incompatible with that preload,
    so isolated bpy workers must start before libOpen3D enters their address
    space. The parent ComfyUI process remains unchanged.
    """

    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    return environment

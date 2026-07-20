"""Install bundled Blender add-ons into the active bpy user add-on path.

This module intentionally does not import ``bpy`` at import time. Comfy3D's Docker
image provides bpy in the runtime environment, while dependency installation for
custom nodes can happen earlier in the image build before bpy is available.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger("unirig.blender_addons")


def _copy_package(source: Path, target: Path) -> None:
    if not (source / "__init__.py").exists():
        raise RuntimeError(f"{source} is not a Blender add-on package directory")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def install_bundled_addon(source: Path, module_name: str) -> Path | None:
    """Copy and enable a bundled Blender add-on.

    Returns the installed path, or ``None`` if bpy is unavailable. Runtime errors
    while enabling are logged and re-raised because a partially installed add-on
    should be visible during startup rather than silently failing.
    """

    try:
        import bpy  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on Blender runtime
        log.warning("bpy is unavailable; skipping Blender add-on %s install: %s", module_name, exc)
        return None

    addon_dir = Path(bpy.utils.user_resource("SCRIPTS", path="addons", create=True))
    target = addon_dir / module_name
    _copy_package(source, target)
    log.info("installed bundled Blender add-on %s at %s", module_name, target)

    if str(addon_dir) not in sys.path:
        sys.path.append(str(addon_dir))

    try:
        bpy.ops.preferences.addon_enable(module=module_name)
        bpy.ops.wm.save_userpref()
    except Exception:
        log.exception("failed to enable bundled Blender add-on %s", module_name)
        raise

    log.info("enabled bundled Blender add-on %s", module_name)
    return target

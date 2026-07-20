"""ComfyUI-UniRig prestartup for the Comfy3D Docker environment."""

from pathlib import Path
import shutil
import sys

from comfy_3d_viewers import copy_viewer


def copy_files(src, dst, pattern="*", overwrite=False):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for file_path in src.glob(pattern):
        if not file_path.is_file():
            continue
        target = dst / file_path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            shutil.copy2(file_path, target)
            copied += 1
    return copied


SCRIPT_DIR = Path(__file__).parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from unirig_utils.install_blender_addon import install_bundled_addon

# Install bundled Blender add-ons
install_bundled_addon(SCRIPT_DIR / "third_party/auto_rig_pro", "auto_rig_pro")

# Copy viewers
copy_viewer("fbx", SCRIPT_DIR / "web")
copy_viewer("fbx_debug", SCRIPT_DIR / "web")
copy_viewer("fbx_compare", SCRIPT_DIR / "web")

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input/3d")
copy_files(SCRIPT_DIR / "assets/animation_templates", COMFYUI_DIR / "input/animation_templates", "**/*")
copy_files(SCRIPT_DIR / "assets/animation_characters", COMFYUI_DIR / "input/animation_characters")

"""Comfy3D fork: dependencies are installed directly from requirements.txt.

The upstream plugin uses comfy-env/pixi isolated environments. Comfy3D builds a
single controlled Docker environment with bpy installed from Michael's buildbpy
wheel index, so there is intentionally no runtime installer here.
"""

print("ComfyUI-UniRig Comfy3D fork: no comfy-env install step required.")

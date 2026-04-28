"""
MIAAutoRig - Fast humanoid rigging using Make-It-Animatable.
"""

import logging
import time
from pathlib import Path
import comfy.utils

from comfy_api.latest import io


def _mm():
    import comfy.model_management
    return comfy.model_management

log = logging.getLogger("unirig")

# ComfyUI folder paths
try:
    import folder_paths
    OUTPUT_DIR = Path(folder_paths.get_output_directory())
except ImportError:
    OUTPUT_DIR = Path(__file__).parent.parent / "output"


class MIAAutoRig(io.ComfyNode):
    """
    Fast humanoid rigging using Make-It-Animatable.

    Takes a mesh and outputs a Mixamo-compatible rigged FBX file.
    Optimized for humanoid characters - faster than UniRig (<1 second).

    Outputs FBX with Mixamo skeleton ready for Mixamo animations.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MIAAutoRig",
            display_name="MIA: Auto Rig",
            category="UniRig/MIA",
            description="Fast humanoid rigging using Make-It-Animatable. Takes a mesh and outputs a Mixamo-compatible rigged FBX file. Optimized for humanoid characters - faster than UniRig (<1 second).",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Custom("MIA_MODEL").Input("model",
                    tooltip="Pre-loaded MIA model (from MIALoadModel)"),
                io.String.Input("fbx_name", default="", optional=True,
                                tooltip="Custom filename for saved FBX (without extension). If empty, uses auto-generated name."),
                io.Boolean.Input("no_fingers", default=True, optional=True,
                                 tooltip="Merge finger weights to hand bone. Enable if model doesn't have separate fingers."),
                io.Boolean.Input("use_normal", default=False, optional=True,
                                 tooltip="Use surface normals for better skinning weights. Helps when limbs are close together."),
                io.Boolean.Input("reset_to_rest", default=True, optional=True,
                                 tooltip="Transform output to T-pose rest position for animation compatibility."),
            ],
            outputs=[
                io.String.Output(display_name="fbx_output_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        model,
        fbx_name="",
        no_fingers=True,
        use_normal=False,
        reset_to_rest=True,
    ):
        """
        Complete rigging pipeline using Make-It-Animatable.

        1. Sample points from mesh surface
        2. Normalize and localize joints (coarse)
        3. Predict blend weights, joint positions, and pose
        4. Post-process and export FBX
        """
        # Lazy import - only run in isolated worker
        from .mia_inference import load_mia_models, get_cached_models, run_mia_inference

        total_start = time.time()

        # DEBUG: Check visual immediately on receipt (before any MIA code)
        log.debug("Received mesh visual type: %s", type(trimesh.visual).__name__ if hasattr(trimesh, 'visual') else 'NO VISUAL')
        if hasattr(trimesh, 'visual') and hasattr(trimesh.visual, 'material'):
            log.debug("  Material: %s", type(trimesh.visual.material).__name__)

        # Progress bar for MIA pipeline steps (load models, inference, export)
        pbar = comfy.utils.ProgressBar(3)

        log.info("Starting Make-It-Animatable rigging pipeline...")
        log.info("Options: no_fingers=%s, use_normal=%s, reset_to_rest=%s", no_fingers, use_normal, reset_to_rest)

        # model is a config dict from MIALoadModel - extract settings
        dtype = model.get("dtype", "fp32")
        log.info("Config: dtype=%s", dtype)

        # Load models internally (downloads from HuggingFace if needed)
        cache_key = load_mia_models(dtype=dtype)
        models = get_cached_models(cache_key)
        pbar.update(1)

        # Check for interruption before inference
        _mm().throw_exception_if_processing_interrupted()

        # Generate output filename
        if fbx_name:
            output_filename = f"{fbx_name}_mia.fbx"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"rigged_mia_{timestamp}.fbx"

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / output_filename)

        # Run MIA inference with loaded models
        result_path = run_mia_inference(
            mesh=trimesh,
            models=models,
            output_path=output_path,
            no_fingers=no_fingers,
            use_normal=use_normal,
            reset_to_rest=reset_to_rest,
        )

        pbar.update(2)

        total_time = time.time() - total_start
        log.info("========================================")
        log.info("Make-It-Animatable rigging complete!")
        log.info("Total time: %.2fs", total_time)
        log.info("Output: %s", result_path)
        log.info("========================================")

        return io.NodeOutput(result_path)

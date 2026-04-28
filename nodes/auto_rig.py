"""
UniRigAutoRig - Single node for complete rigging pipeline.
Takes mesh, outputs animation-ready FBX.
"""

import logging
import os
import sys
import time

import numpy as np
import comfy.utils

from comfy_api.latest import io


def _mm():
    import comfy.model_management
    return comfy.model_management

log = logging.getLogger("unirig")

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .skeleton_extraction import UniRigExtractSkeletonNew
    from .skinning import UniRigApplySkinningMLNew
except ImportError:
    from skeleton_extraction import UniRigExtractSkeletonNew
    from skinning import UniRigApplySkinningMLNew


class UniRigAutoRig(io.ComfyNode):
    """
    Single node for complete rigging pipeline.

    Combines skeleton extraction + skinning + normalization into one step.
    Takes mesh, outputs animation-ready FBX.

    Runs in isolated environment with GPU dependencies.
    For Mixamo template: outputs FBX normalized to Mixamo rest pose
    (T-pose, human scale, Hips at 1.04m) ready for Mixamo animations.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniRigAutoRig",
            display_name="UniRig: Auto Rig",
            category="UniRig",
            description="Single node for complete rigging pipeline. Combines skeleton extraction + skinning + normalization into one step. Takes mesh, outputs animation-ready FBX.",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Custom("UNIRIG_MODEL").Input("model",
                    tooltip="Pre-loaded UniRig model (from UniRigLoadModel)"),
                io.Combo.Input("skeleton_template", options=["mixamo", "articulationxl"],
                               default="mixamo", optional=True,
                               tooltip="Skeleton template. 'mixamo' remaps to Mixamo bone names (humanoids). 'articulationxl' outputs native skeleton (any 3D asset)."),
                io.String.Input("fbx_name", default="", optional=True,
                                tooltip="Custom filename for saved FBX (without extension). If empty, uses auto-generated name."),
                io.Int.Input("target_face_count", default=50000, min=10000, max=500000, step=10000,
                             optional=True,
                             tooltip="Target face count for mesh decimation. Warning: changing from default may reduce quality."),
                io.Boolean.Input("debug", default=False, optional=True,
                                 tooltip="Enable detailed debug logging for the entire rigging pipeline."),
            ],
            outputs=[
                io.String.Output(display_name="fbx_output_path"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, model,
                skeleton_template="mixamo", fbx_name="", target_face_count=50000,
                debug=False):
        """
        Complete rigging pipeline in one step.

        1. Extract skeleton from mesh
        2. Compute skin weights
        3. Normalize for target template (happens in blender_export_fbx.py)
        4. Export FBX
        """
        if debug:
            logging.getLogger("unirig").setLevel(logging.DEBUG)

        total_start = time.time()
        log.info("Starting complete rigging pipeline...")
        log.info("Skeleton template: %s", skeleton_template)
        log.debug("Model keys: %s", list(model.keys()))
        log.debug("Model dtype: %s, attn_backend: %s", model.get("dtype"), model.get("attn_backend"))

        # Progress bar for the 2-step pipeline (skeleton extraction + skinning)
        pbar = comfy.utils.ProgressBar(2)

        # Extract individual models from combined model
        skeleton_model = model["skeleton_model"]
        skinning_model = model["skinning_model"]

        # Propagate dtype and attn_backend if not already set
        for sub_model in (skeleton_model, skinning_model):
            if "dtype" not in sub_model:
                sub_model["dtype"] = model.get("dtype")
            if "attn_backend" not in sub_model:
                sub_model["attn_backend"] = model.get("attn_backend", "auto")

        # Step 1: Extract skeleton
        log.info("Step 1/2: Extracting skeleton...")
        import torch
        dev = _mm().get_torch_device()
        if dev.type == 'cuda':
            a = torch.cuda.memory_allocated(dev) / (1024**3)
            r = torch.cuda.memory_reserved(dev) / (1024**3)
            log.debug("[VRAM AutoRig] before skeleton extraction: allocated=%.2fGB reserved=%.2fGB", a, r)
        step_start = time.time()

        normalized_mesh, skeleton, texture_preview = UniRigExtractSkeletonNew.execute(
            trimesh=trimesh,
            skeleton_model=skeleton_model,
            seed=42,  # Fixed seed for reproducibility
            skeleton_template=skeleton_template,
            target_face_count=target_face_count
        )

        skeleton_time = time.time() - step_start
        log.info("Skeleton extraction complete in %.2fs", skeleton_time)
        log.info("Extracted %d bones", len(skeleton.get('names', [])))
        log.debug("Bone names: %s", skeleton.get('names', []))
        log.debug("Bone parents: %s", skeleton.get('parents', []))
        log.debug("Joints shape: %s", np.array(skeleton.get('joints', [])).shape if skeleton.get('joints') is not None else None)
        log.debug("Edges: %d", len(skeleton.get('edges', [])))
        pbar.update(1)

        # Check for interruption before skinning
        _mm().throw_exception_if_processing_interrupted()

        # Step 2: Apply skinning
        log.info("Step 2/2: Applying skinning...")
        if dev.type == 'cuda':
            a = torch.cuda.memory_allocated(dev) / (1024**3)
            r = torch.cuda.memory_reserved(dev) / (1024**3)
            log.debug("[VRAM AutoRig] before skinning: allocated=%.2fGB reserved=%.2fGB", a, r)
        step_start = time.time()

        fbx_output_path, _ = UniRigApplySkinningMLNew.execute(
            normalized_mesh=normalized_mesh,
            skeleton=skeleton,
            skinning_model=skinning_model,
            fbx_name=fbx_name,
            voxel_grid_size=196,      # Model trained with this
            num_samples=32768,         # Optimal default
            vertex_samples=8192,       # Optimal default
            voxel_mask_power=0.5       # Model trained with this
        )

        skinning_time = time.time() - step_start
        log.info("Skinning complete in %.2fs", skinning_time)
        log.debug("FBX output: %s", fbx_output_path)
        pbar.update(1)

        total_time = time.time() - total_start
        log.info("========================================")
        log.info("Complete rigging pipeline finished!")
        log.info("Total time: %.2fs", total_time)
        log.info("Output: %s", fbx_output_path)
        log.info("========================================")

        return io.NodeOutput(fbx_output_path)

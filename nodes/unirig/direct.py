"""
Direct inference for UniRig models - no subprocess, no Lightning Trainer.

This module provides simple Python functions to run inference directly,
keeping models loaded in memory for fast repeated inference.

Uses ComfyUI ModelPatcher for proper memory management.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging
import comfy.model_patcher
import comfy.ops
import comfy.utils

log = logging.getLogger("unirig")


def _mm():
    import comfy.model_management
    return comfy.model_management

# Local model cache for the worker process. Keeps loaded models + ModelPatchers
# in memory across prompt executions within the same persistent worker.
_loaded_models = {}


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface using barycentric coordinates.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Face indices (F, 3)
        num_samples: Number of points to sample
        seed: Random seed

    Returns:
        Tuple of (sampled_points, sampled_normals) each (num_samples, 3)
    """
    rng = np.random.RandomState(seed)

    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product gives 2x area
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) / 2

    # Normalize face normals
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)

    # Sample faces weighted by area
    probs = face_areas / face_areas.sum()
    face_indices = rng.choice(len(faces), size=num_samples, p=probs)

    # Sample barycentric coordinates
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)

    # Ensure point is inside triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Get sampled points
    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]

    points = (
        u[:, None] * sampled_v0 +
        v[:, None] * sampled_v1 +
        w[:, None] * sampled_v2
    )

    # Get normals for sampled faces
    normals = face_normals[face_indices]

    return points.astype(np.float32), normals.astype(np.float32)


def normalize_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize vertices to [-1, 1] range.

    Args:
        vertices: Input vertices (N, 3)

    Returns:
        Tuple of (normalized_vertices, normalization_params)
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        normalized = (vertices - center) / scale
    else:
        normalized = vertices - center

    params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords,
    }

    return normalized.astype(np.float32), params


def _get_device():
    """Get the best available device."""
    return _mm().get_torch_device()


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

def _resolve_dtype(dtype):
    """Convert string dtype (e.g. 'bf16') to torch.dtype, pass through if already a dtype."""
    if isinstance(dtype, str):
        return _DTYPE_MAP.get(dtype, torch.float32)
    return dtype


def _fix_leftover_meta_tensors(model):
    """Replace any leftover meta-device parameters/buffers with real CPU tensors.

    After constructing a model on torch.device('meta') and loading weights with
    assign=True, registered buffers that are not present in the state_dict remain
    on the meta device.  This helper materialises them on CPU so the model is
    fully usable.
    """
    # Fix meta buffers (e.g. FrequencyPositionalEmbedding.frequencies)
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")

    # Fix meta parameters (shouldn't happen if state_dict is complete, but be safe)
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._parameters[parts[-1]] = torch.nn.Parameter(
                torch.zeros_like(param, device="cpu"),
                requires_grad=param.requires_grad,
            )


def _load_skeleton_model(checkpoint_path: str, dtype=None, attn_backend: str = "auto"):
    """
    Load the skeleton (AR) model directly.

    Uses meta-device initialization to avoid 2x RAM during model construction:
    the model is first built on torch.device('meta') (zero memory), then real
    weights are loaded from disk and assigned in-place.

    Args:
        checkpoint_path: Path to skeleton.safetensors
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        (model, tokenizer) tuple - model on CPU with requested dtype
    """
    from .tokenizer_parse import get_tokenizer
    from .tokenizer_spec import TokenizerConfig
    from .model_parse import get_model
    from . import unirig_ar
    from .configs import AR_MODEL_CONFIG, TOKENIZER_CONFIG

    # Override attention backend before model creation
    if attn_backend == "sdpa":
        unirig_ar.FLASH_ATTN_AVAILABLE = False
        log.info("Attention backend: sdpa (flash_attn disabled)")
    elif attn_backend == "flash_attn":
        if not unirig_ar.FLASH_ATTN_AVAILABLE:
            raise RuntimeError("flash_attn requested but not installed")
        log.info("Attention backend: flash_attn")
    else:
        log.info("Attention backend: auto (flash_attn=%s)", unirig_ar.FLASH_ATTN_AVAILABLE)

    # Resolve dtype and pick ComfyUI operations
    model_dtype = _resolve_dtype(dtype)
    if model_dtype is not None:
        operations = comfy.ops.pick_operations(model_dtype, model_dtype)
    else:
        operations = comfy.ops.disable_weight_init
    log.info("Skeleton ops: %s (dtype=%s)", operations, model_dtype)

    # Build tokenizer from inlined config
    tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=TOKENIZER_CONFIG))

    # Build model on meta device (zero RAM) to avoid 2x memory during construction
    log.info("Building skeleton model on meta device (zero RAM allocation)...")
    model_config = dict(AR_MODEL_CONFIG)
    model_config['tokenizer'] = tokenizer
    model_config['dtype'] = model_dtype
    model_config['device'] = 'meta'
    model_config['operations'] = operations
    with torch.device("meta"):
        model = get_model(**model_config)

    # Load weights with comfy.utils (safe, memory-efficient) and assign directly
    log.info("Loading skeleton weights from %s", checkpoint_path)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # assign=True: directly replace meta tensors with loaded tensors (no copy)
    result = model.load_state_dict(cleaned_state_dict, strict=False, assign=True)
    log.debug("Skeleton load_state_dict: missing=%d, unexpected=%d", len(result.missing_keys), len(result.unexpected_keys))
    if result.missing_keys:
        log.debug("  Missing keys: %s", result.missing_keys[:20])
    if result.unexpected_keys:
        log.debug("  Unexpected keys: %s", result.unexpected_keys[:20])

    # Fix any leftover meta-device buffers not present in state_dict
    meta_count = sum(1 for _, b in model.named_buffers() if b.device.type == "meta")
    log.debug("Leftover meta buffers before fix: %d", meta_count)
    _fix_leftover_meta_tensors(model)
    meta_count_after = sum(1 for _, b in model.named_buffers() if b.device.type == "meta")
    log.debug("Leftover meta buffers after fix: %d", meta_count_after)

    model.eval()

    # Debug: verify frequency buffers are non-zero
    for name, buf in model.named_buffers():
        if "frequencies" in name:
            log.debug("Buffer %s: shape=%s, device=%s, first_5=%s", name, buf.shape, buf.device, buf[:5].tolist() if buf.numel() >= 5 else buf.tolist())

    # Safety net: ensure all weights match requested dtype
    if model_dtype is not None:
        model = model.to(dtype=model_dtype)
        log.info("Skeleton model dtype: %s", model_dtype)

    param_count = sum(p.numel() for p in model.parameters())
    log.debug("Skeleton model params: %d (%.1f M)", param_count, param_count / 1e6)
    log.info("Skeleton model ready (on CPU, will be moved to GPU by ModelPatcher)")
    return model, tokenizer


def _load_skin_model(checkpoint_path: str, dtype=None, attn_backend: str = "auto"):
    """
    Load the skinning model directly.

    Uses meta-device initialization to avoid 2x RAM during model construction:
    the model is first built on torch.device('meta') (zero memory), then real
    weights are loaded from disk and assigned in-place.

    Args:
        checkpoint_path: Path to skin.safetensors
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Loaded model on CPU with requested dtype
    """
    from .model_parse import get_model
    from .configs import SKIN_MODEL_CONFIG

    log.info("Skin attention backend: ComfyUI optimized_attention")

    # Resolve dtype and pick ComfyUI operations
    model_dtype = _resolve_dtype(dtype)
    if model_dtype is not None:
        operations = comfy.ops.pick_operations(model_dtype, model_dtype)
    else:
        operations = comfy.ops.disable_weight_init
    log.info("Skin ops: %s (dtype=%s)", operations, model_dtype)

    # Build model on meta device (zero RAM) to avoid 2x memory during construction
    log.info("Building skin model on meta device (zero RAM allocation)...")
    model_config = dict(SKIN_MODEL_CONFIG)
    model_config['dtype'] = model_dtype
    model_config['device'] = 'meta'
    model_config['operations'] = operations
    with torch.device("meta"):
        model = get_model(tokenizer=None, **model_config)

    # Load weights with comfy.utils (safe, memory-efficient) and assign directly
    log.info("Loading skin weights from %s", checkpoint_path)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # assign=True: directly replace meta tensors with loaded tensors (no copy)
    result = model.load_state_dict(cleaned_state_dict, strict=False, assign=True)
    log.debug("Skin load_state_dict: missing=%d, unexpected=%d", len(result.missing_keys), len(result.unexpected_keys))
    if result.missing_keys:
        log.debug("  Missing keys: %s", result.missing_keys[:20])
    if result.unexpected_keys:
        log.debug("  Unexpected keys: %s", result.unexpected_keys[:20])

    # Fix any leftover meta-device buffers not present in state_dict
    meta_count = sum(1 for _, b in model.named_buffers() if b.device.type == "meta")
    log.debug("Leftover meta buffers before fix: %d", meta_count)
    _fix_leftover_meta_tensors(model)
    meta_count_after = sum(1 for _, b in model.named_buffers() if b.device.type == "meta")
    log.debug("Leftover meta buffers after fix: %d", meta_count_after)

    model.eval()

    # Debug: verify frequency buffers are non-zero
    for name, buf in model.named_buffers():
        if "frequencies" in name:
            log.debug("Buffer %s: shape=%s, device=%s, first_5=%s", name, buf.shape, buf.device, buf[:5].tolist() if buf.numel() >= 5 else buf.tolist())

    # Safety net: ensure all weights match requested dtype
    if model_dtype is not None:
        model = model.to(dtype=model_dtype)
        log.info("Skin model dtype: %s", model_dtype)

    param_count = sum(p.numel() for p in model.parameters())
    log.debug("Skin model params: %d (%.1f M)", param_count, param_count / 1e6)
    log.info("Skin model ready (on CPU, will be moved to GPU by ModelPatcher)")
    return model


def get_skeleton_model(
    checkpoint_path: str,
    dtype=None,
    attn_backend: str = "auto",
    force_reload: bool = False,
):
    """Get or load cached skeleton model wrapped in ModelPatcher.

    Returns:
        (ModelPatcher, tokenizer) tuple
    """
    cache = _loaded_models
    cache_key = f"skeleton:{checkpoint_path}:{dtype}"
    if cache_key not in cache or force_reload:
        model, tokenizer = _load_skeleton_model(
            checkpoint_path, dtype=dtype, attn_backend=attn_backend
        )
        load_device = _mm().get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        cache[cache_key] = (patcher, tokenizer)
        log.info("Skeleton model cached with ModelPatcher (load=%s, offload=%s)", load_device, offload_device)
    return cache[cache_key]


def get_skin_model(
    checkpoint_path: str,
    dtype=None,
    attn_backend: str = "auto",
    force_reload: bool = False,
):
    """Get or load cached skin model wrapped in ModelPatcher.

    Returns:
        ModelPatcher instance
    """
    cache = _loaded_models
    cache_key = f"skin:{checkpoint_path}:{dtype}"
    if cache_key not in cache or force_reload:
        model = _load_skin_model(
            checkpoint_path, dtype=dtype, attn_backend=attn_backend
        )
        load_device = _mm().get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        cache[cache_key] = patcher
        log.info("Skin model cached with ModelPatcher (load=%s, offload=%s)", load_device, offload_device)
    return cache[cache_key]


@torch.no_grad()
def predict_skeleton(
    vertices: np.ndarray,
    normals: np.ndarray,
    checkpoint_path: str,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    dtype=None,
    attn_backend: str = "auto",
    **generate_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Predict skeleton from mesh vertices and normals.

    Uses ComfyUI ModelPatcher for GPU memory management.

    Args:
        vertices: Mesh vertices (N, 3), should be normalized to [-1, 1]
        normals: Vertex normals (N, 3)
        checkpoint_path: Path to skeleton.safetensors
        cls: Class name for conditioning (default: "articulationxl")
        max_new_tokens: Maximum tokens to generate
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')
        **generate_kwargs: Additional kwargs for generation (num_beams, etc.)

    Returns:
        Dict with 'joints', 'parents', 'names' etc from detokenization
    """
    patcher, tokenizer = get_skeleton_model(
        checkpoint_path, dtype=dtype, attn_backend=attn_backend
    )

    # Let ComfyUI manage GPU memory — beam search (15 beams × 2048 tokens) needs ~2 GB
    memory_required = 2 * 1024 * 1024 * 1024
    _mm().load_models_gpu([patcher], memory_required=memory_required)
    device = patcher.load_device

    if device.type == 'cuda':
        a = torch.cuda.memory_allocated(device) / (1024**3)
        r = torch.cuda.memory_reserved(device) / (1024**3)
        log.debug("[VRAM predict_skeleton] after load_models_gpu: allocated=%.2fGB reserved=%.2fGB", a, r)

    # Convert to tensors in the model's dtype
    model_dtype = patcher.model.get_dtype() or torch.float32
    log.debug("[VRAM predict_skeleton] vertices=%s, normals=%s, dtype=%s, device=%s, cls=%s",
              vertices.shape, normals.shape, model_dtype, device, cls)
    vertices_t = torch.from_numpy(vertices).to(dtype=model_dtype, device=device)
    normals_t = torch.from_numpy(normals).to(dtype=model_dtype, device=device)

    # Default generation kwargs (matching ar_inference_articulationxl.yaml)
    default_kwargs = {
        'max_new_tokens': max_new_tokens,
        'num_beams': 15,
        'do_sample': True,
        'top_k': 5,
        'top_p': 0.95,
        'repetition_penalty': 3.0,
        'temperature': 1.5,
    }
    default_kwargs.update(generate_kwargs)
    log.debug("Generation kwargs: %s", default_kwargs)

    # Run generation
    if device.type == 'cuda':
        a = torch.cuda.memory_allocated(device) / (1024**3)
        r = torch.cuda.memory_reserved(device) / (1024**3)
        log.debug("[VRAM predict_skeleton] before generate: allocated=%.2fGB reserved=%.2fGB", a, r)
    model = patcher.model
    result = model.generate(
        vertices=vertices_t,
        normals=normals_t,
        cls=cls,
        **default_kwargs,
    )

    # Release fragmented CUDA blocks from beam search before skinning runs
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Convert result to dict of numpy arrays
    output = {
        'joints': np.array(result.joints) if result.joints is not None else None,
        'parents': np.array(result.parents) if result.parents is not None else None,
        'names': result.names if hasattr(result, 'names') else None,
    }

    # Add any other attributes from DetokenizeOutput
    for attr in ['tails', 'bone_names', 'root_idx']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                output[attr] = np.array(val) if isinstance(val, (list, tuple)) else val

    log.debug("predict_skeleton output: joints=%s, parents=%s, names=%s",
              output['joints'].shape if output['joints'] is not None else None,
              output['parents'].shape if output['parents'] is not None else None,
              output['names'][:5] if output['names'] is not None else None)
    return output


@torch.no_grad()
def predict_skinning(
    vertices: np.ndarray,
    normals: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    checkpoint_path: str,
    faces: Optional[np.ndarray] = None,
    tails: Optional[np.ndarray] = None,
    voxel_grid_size: int = 196,
    dtype=None,
    attn_backend: str = "auto",
) -> np.ndarray:
    """
    Predict skinning weights from mesh and skeleton.

    Uses ComfyUI ModelPatcher for GPU memory management.

    Args:
        vertices: Mesh vertices (N, 3)
        normals: Vertex normals (N, 3)
        joints: Joint positions (J, 3)
        parents: Parent indices (J,)
        checkpoint_path: Path to skin.safetensors
        faces: Mesh faces (F, 3) - required for voxel_skin computation
        tails: Bone tail positions (J, 3) - if None, computed from joints
        voxel_grid_size: Grid size for voxel_skin (default 196)
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Skinning weights (N, J)
    """
    patcher = get_skin_model(
        checkpoint_path, dtype=dtype, attn_backend=attn_backend
    )

    # Let ComfyUI manage GPU memory — skinning forward pass needs ~1 GB
    memory_required = 1 * 1024 * 1024 * 1024
    _mm().load_models_gpu([patcher], memory_required=memory_required)
    device = patcher.load_device

    if device.type == 'cuda':
        a = torch.cuda.memory_allocated(device) / (1024**3)
        r = torch.cuda.memory_reserved(device) / (1024**3)
        log.debug("[VRAM predict_skinning] after load_models_gpu: allocated=%.2fGB reserved=%.2fGB", a, r)

    num_joints = len(joints)
    num_vertices = len(vertices)
    log.debug("[VRAM predict_skinning] vertices=%s, normals=%s, joints=%s, parents=%s, dtype=%s, device=%s",
              vertices.shape, normals.shape, joints.shape, parents.shape, dtype, device)

    # Compute tails if not provided (use joint positions as tails)
    if tails is None:
        # Default: tails = joints (bone has zero length, point bone)
        tails = joints.copy()

    # Compute voxel_skin if faces provided
    if faces is not None:
        # Import voxelization functions
        from .data_vertex_group import voxelization, voxel_skin

        log.info("Computing voxel_skin (grid=%s)...", voxel_grid_size)

        # Voxelize the mesh
        grid_coords = voxelization(
            vertices=vertices,
            faces=faces,
            grid=voxel_grid_size,
            backend='trimesh',
        )

        # Compute voxel skin weights
        voxel_skin_weights = voxel_skin(
            grid=voxel_grid_size,
            grid_coords=grid_coords,
            joints=joints,
            vertices=vertices,
            faces=faces,
            alpha=0.5,
            link_dis=0.00001,
            grid_query=7,
            vertex_query=1,
            grid_weight=3.0,
        )
        voxel_skin_weights = np.nan_to_num(voxel_skin_weights, nan=0., posinf=0., neginf=0.)
        log.info("voxel_skin shape: %s", voxel_skin_weights.shape)
        log.debug("voxel_skin range: min=%.6f, max=%.6f, mean=%.6f", voxel_skin_weights.min(), voxel_skin_weights.max(), voxel_skin_weights.mean())
    else:
        # Fallback: zero voxel_skin
        voxel_skin_weights = np.zeros((num_joints, num_vertices), dtype=np.float32)

    # Prepare batch in the model's dtype
    model_dtype = patcher.model.get_dtype() or torch.float32
    log.debug("Skin model dtype: %s", model_dtype)
    batch = {
        'vertices': torch.from_numpy(vertices).to(dtype=model_dtype).unsqueeze(0).to(device),
        'normals': torch.from_numpy(normals).to(dtype=model_dtype).unsqueeze(0).to(device),
        'joints': torch.from_numpy(joints).to(dtype=model_dtype).unsqueeze(0).to(device),
        'tails': torch.from_numpy(tails).to(dtype=model_dtype).unsqueeze(0).to(device),
        'voxel_skin': torch.from_numpy(voxel_skin_weights).to(dtype=model_dtype).unsqueeze(0).to(device),
        'parents': torch.from_numpy(parents).long().unsqueeze(0).to(device),
        'num_bones': torch.tensor([num_joints], dtype=torch.long, device=device),
        'offset': torch.tensor([num_vertices], dtype=torch.long, device=device),  # PTv3 batch offset for single sample
        'path': ['direct_inference'],
    }

    # Debug: log batch tensor shapes
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            log.debug("Batch[%s]: shape=%s, dtype=%s, device=%s", k, v.shape, v.dtype, v.device)

    # Run prediction
    model = patcher.model
    model.to(device)
    result = model.predict_step(batch)

    # Extract skin weights - result is a list of tensors, one per batch
    skin_weights = result[0]  # Get first (only) batch item
    if isinstance(skin_weights, torch.Tensor):
        skin_weights = skin_weights.cpu().float().numpy()

    log.debug("predict_skinning output: shape=%s, min=%.6f, max=%.6f", skin_weights.shape, skin_weights.min(), skin_weights.max())
    return skin_weights


@torch.no_grad()
def predict_skeleton_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    seed: int = 42,
    dtype=None,
    attn_backend: str = "auto",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    High-level function: predict skeleton from raw mesh.

    Handles all preprocessing (normalization, sampling) internally.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for sampling
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Tuple of (skeleton_dict, normalization_params)
        - skeleton_dict: Dict with 'joints', 'parents', 'names', etc.
        - normalization_params: Dict with 'center', 'scale' for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)
    log.debug("Normalized vertices: shape=%s, range=[%.3f, %.3f]", norm_vertices.shape, norm_vertices.min(), norm_vertices.max())
    log.debug("Normalization: center=%s, scale=%.4f", norm_params['center'], norm_params['scale'])

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )
    log.debug("Sampled %d surface points, normals shape=%s", len(sampled_points), sampled_normals.shape)

    # 3. Run skeleton prediction
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    return skeleton, norm_params


@torch.no_grad()
def rig_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    skin_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    seed: int = 42,
    dtype=None,
    attn_backend: str = "auto",
) -> Dict[str, Any]:
    """
    Complete rigging pipeline: mesh -> skeleton -> skinning weights.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        skin_checkpoint: Path to skin.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        seed: Random seed
        dtype: Model precision (torch.bfloat16, torch.float16, etc.)
        attn_backend: Attention backend ('auto', 'flash_attn', 'sdpa')

    Returns:
        Dict with:
            - 'joints': Joint positions (J, 3) in normalized space
            - 'parents': Parent indices (J,)
            - 'names': Joint names
            - 'skin_weights': Skinning weights (num_samples, J)
            - 'sampled_vertices': Sampled surface points (num_samples, 3)
            - 'sampled_normals': Sampled normals (num_samples, 3)
            - 'normalization': Dict with center/scale for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )

    # 3. Predict skeleton
    log.info("Predicting skeleton...")
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    if skeleton['joints'] is None:
        raise RuntimeError("Skeleton prediction failed - no joints generated")

    # 4. Predict skinning weights
    log.info("Predicting skinning weights...")
    skin_weights = predict_skinning(
        vertices=sampled_points,
        normals=sampled_normals,
        joints=skeleton['joints'],
        parents=skeleton['parents'],
        checkpoint_path=skin_checkpoint,
        dtype=dtype,
        attn_backend=attn_backend,
    )

    return {
        'joints': skeleton['joints'],
        'parents': skeleton['parents'],
        'names': skeleton.get('names'),
        'tails': skeleton.get('tails'),
        'skin_weights': skin_weights,
        'sampled_vertices': sampled_points,
        'sampled_normals': sampled_normals,
        'normalization': norm_params,
    }

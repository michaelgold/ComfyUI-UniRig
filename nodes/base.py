"""
Base setup and shared utilities for UniRig nodes.

Handles path configuration, Blender setup, and HuggingFace cache management.
"""

import logging
import os
from pathlib import Path
import numpy as np
import base64
from io import BytesIO

import folder_paths

log = logging.getLogger("unirig")

# Try to import PIL for texture handling
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    log.warning("PIL not available, texture preview will be limited")


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.parent.absolute()  # Go up from nodes/ to ComfyUI-UniRig/
NODES_DIR = Path(__file__).parent.absolute()  # nodes/ directory itself
UNIRIG_PATH = str(NODES_DIR / "unirig")
# Keep LIB_DIR for backwards compatibility
LIB_DIR = NODES_DIR

# Set up UniRig models directory in ComfyUI's models folder
# Only contains skeleton.safetensors and skin.safetensors - no HuggingFace cache
UNIRIG_MODELS_DIR = Path(folder_paths.models_dir) / "unirig"
UNIRIG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
folder_paths.add_model_folder_path("unirig", str(UNIRIG_MODELS_DIR))

log.info("Models directory: %s", UNIRIG_MODELS_DIR)

import shutil


def decode_texture_to_comfy_image(texture_data_base64: str):
    """
    Decode base64 texture to ComfyUI IMAGE format (torch tensor).

    Args:
        texture_data_base64: Base64-encoded image data

    Returns:
        tuple: (torch tensor [1, H, W, 3], width, height) or (None, 0, 0)
    """
    if not texture_data_base64 or not HAS_PIL:
        return None, 0, 0

    try:
        import torch  # Lazy import

        # Decode base64
        image_data = base64.b64decode(texture_data_base64)
        pil_image = PILImage.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to torch tensor [1, H, W, 3] for ComfyUI
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor, pil_image.width, pil_image.height

    except Exception as e:
        log.error("Error decoding texture: %s", e)
        return None, 0, 0


def create_placeholder_texture(width: int = 256, height: int = 256, text: str = "No Texture"):
    """
    Create a placeholder image when no texture is available.

    Args:
        width: Image width
        height: Image height
        text: Text to display (not currently rendered, just for reference)

    Returns:
        torch.Tensor: Placeholder image tensor [1, H, W, 3]
    """
    import torch  # Lazy import

    try:
        # Create a gray image with text
        img_array = np.full((height, width, 3), 0.3, dtype=np.float32)

        # Add a simple pattern to indicate placeholder
        # Create a grid pattern
        for i in range(0, height, 32):
            img_array[i:i+2, :, :] = 0.4
        for j in range(0, width, 32):
            img_array[:, j:j+2, :] = 0.4

        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    except Exception as e:
        log.error("Error creating placeholder: %s", e)
        # Return minimal gray image
        return torch.full((1, 64, 64, 3), 0.3)

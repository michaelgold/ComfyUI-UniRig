import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

log = logging.getLogger("unirig")

def _get_models_dir() -> Path:
    """Get the UniRig models directory via ComfyUI's folder_paths."""
    import folder_paths
    models_dir = Path(folder_paths.models_dir) / "unirig"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def download(ckpt_name: str) -> str:
    """Download model checkpoint, returns path to local file."""
    MAP = {
        'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton.safetensors',
        'experiments/skin/articulation-xl/model.ckpt': 'skin.safetensors',
        'experiments/skin/skeleton/model.ckpt': 'skin.safetensors',
    }

    if ckpt_name not in MAP:
        log.info("Unknown checkpoint: %s", ckpt_name)
        return ckpt_name

    filename = MAP[ckpt_name]
    models_dir = _get_models_dir()
    local_path = models_dir / filename

    # Check if already exists
    if local_path.exists():
        log.info("Found model: %s", local_path)
        return str(local_path)

    try:
        # Create directory if needed
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download directly to models folder
        log.info("Downloading %s from apozz/UniRig-safetensors...", filename)
        hf_hub_download(
            repo_id='apozz/UniRig-safetensors',
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )

        log.info("Downloaded to: %s", local_path)
    except Exception as e:
        log.warning("Failed to download %s: %s", ckpt_name, e)

    # Always return the expected local safetensors path
    return str(local_path)

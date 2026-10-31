"""File-only SkinTokens adapter; no ML or Blender imports in ComfyUI."""
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import tempfile

import folder_paths


def _inside(path, roots):
    return any(path.is_relative_to(root) for root in roots)


def _input_path(value):
    roots = [Path(getter()).resolve() for getter in (
        folder_paths.get_input_directory, folder_paths.get_output_directory,
        folder_paths.get_temp_directory)]
    path = Path(value)
    if not path.is_absolute():
        if '..' in path.parts or '\\' in value:
            raise ValueError('Input traversal is not allowed')
        path = roots[0] / path
    path = path.resolve(strict=True)
    if not _inside(path, roots) or not path.is_file() or path.suffix.lower() != '.glb':
        raise ValueError('Input must be a GLB inside Comfy input, output, or temp')
    return path


def _output_path(value):
    relative = Path(value)
    if (not value or relative.is_absolute() or '..' in relative.parts
            or '\\' in value or ':' in value or relative.suffix.lower() != '.glb'):
        raise ValueError('output_name must be a relative .glb path without traversal')
    root = Path(folder_paths.get_output_directory()).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Output escapes Comfy output directory')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Output already exists; choose a new output_name: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, root


def _validate_glb(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('SkinTokens did not create a fresh GLB output')
    with path.open('rb') as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise RuntimeError('SkinTokens output is empty or truncated')
    magic, version, length = struct.unpack('<4sII', header)
    if magic != b'glTF' or version != 2 or length != path.stat().st_size or length < 20:
        raise RuntimeError('SkinTokens output is not a valid GLB 2 container')


def _run(command, root, timeout):
    env = os.environ.copy()
    # Do not inject Comfy/Open3D libraries into the separate runtime.
    for key in ('LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Includes upstream Blender and dataloader descendants, even on failure.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        log.seek(0, os.SEEK_END)
        log.seek(max(0, log.tell() - 16384))
        detail = log.read().decode('utf-8', errors='replace')
        if timed_out:
            raise RuntimeError(f'SkinTokens timed out after {timeout:g}s\n{detail}')
        if process.returncode:
            raise RuntimeError(f'SkinTokens failed (exit {process.returncode})\n{detail}')


class SkinTokensRigGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'glb_path': ('STRING', {'default': ''}),
            'output_name': ('STRING', {'default': 'skintokens_rigged.glb'}),
        }, 'optional': {
            'use_skeleton': ('BOOLEAN', {'default': False}),
            'use_transfer': ('BOOLEAN', {'default': True}),
            'use_postprocess': ('BOOLEAN', {'default': False}),
            'num_beams': ('INT', {'default': 10, 'min': 1, 'max': 20}),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('glb_path',)
    FUNCTION = 'rig'
    CATEGORY = 'UniRig/SkinTokens'
    OUTPUT_NODE = True

    def rig(self, glb_path, output_name='skintokens_rigged.glb',
            use_skeleton=False, use_transfer=True, use_postprocess=False, num_beams=10):
        if os.name != 'posix':
            raise RuntimeError('SkinTokens adapter requires POSIX process groups (Linux/macOS)')
        if type(num_beams) is not int or not 1 <= num_beams <= 20:
            raise ValueError('num_beams must be an integer from 1 to 20')
        if any(type(flag) is not bool for flag in (use_skeleton, use_transfer, use_postprocess)):
            raise ValueError('SkinTokens options must be booleans')
        source = _input_path(glb_path)
        destination, output_root = _output_path(output_name)
        root_value = os.environ.get('SKINTOKENS_ROOT')
        python_value = os.environ.get('SKINTOKENS_PYTHON')
        if not root_value or not python_value:
            raise RuntimeError('Set SKINTOKENS_ROOT and SKINTOKENS_PYTHON; see docs/skintokens.md')
        root = Path(root_value).expanduser().resolve(strict=True)
        if not all((root / name).is_file() for name in ('demo.py', 'bpy_server.py')):
            raise RuntimeError('SKINTOKENS_ROOT must contain demo.py and bpy_server.py')
        python = shutil.which(python_value)
        if not python:
            raise RuntimeError('SKINTOKENS_PYTHON must name an executable Python interpreter')
        python = os.path.abspath(python)  # preserve venv symlink, do not resolve it
        timeout = float(os.environ.get('SKINTOKENS_TIMEOUT_SECONDS', '1800'))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('SKINTOKENS_TIMEOUT_SECONDS must be finite and positive')
        with tempfile.TemporaryDirectory(prefix='.skintokens-', dir=destination.parent) as staging:
            fresh = Path(staging) / 'rigged.glb'
            command = [python, '-u', str(Path(__file__).with_name('skintokens_worker.py')),
                       '--input', str(source), '--output', str(fresh), '--num_beams', str(num_beams)]
            for name, enabled in (('use_skeleton', use_skeleton), ('use_transfer', use_transfer),
                                  ('use_postprocess', use_postprocess)):
                if enabled:
                    command.append('--' + name)
            _run(command, root, timeout)
            _validate_glb(fresh)
            if not destination.resolve().is_relative_to(output_root):
                raise ValueError('Output escapes Comfy output directory')
            # Atomic no-clobber publication: old/parallel outputs cannot look like success.
            os.link(fresh, destination)
        relative = destination.relative_to(output_root)
        return {'result': (str(destination),), 'ui': {'files': [{
            'filename': relative.name,
            'subfolder': '' if relative.parent == Path('.') else relative.parent.as_posix(),
            'type': 'output',
        }]}}

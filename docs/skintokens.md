# SkinTokens: Rig GLB (optional runtime)

`SkinTokensRigGLB` adds a file-to-file rigging node without replacing any UniRig or MIA nodes. It runs a **separate interpreter**, with its working directory set to `SKINTOKENS_ROOT`. Neither SkinTokens nor `bpy` is imported into ComfyUI by this adapter.

## Install the upstream runtime

The adapter targets upstream `VAST-AI-Research/SkinTokens` commit
`273b691d35989d71cd17ff2895fdc735097b92d1`. Its `demo.py` exposes `run_cli`, uses a separate Blender HTTP server, and defaults to `experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`.

On a Linux NVIDIA host, follow that revision's README prerequisites (Python >=3.11, CUDA toolkit >=12.1, GPU with at least 14 GB memory). Example matching its documented PyTorch/CUDA environment:

```bash
git clone https://github.com/VAST-AI-Research/SkinTokens.git /opt/SkinTokens
cd /opt/SkinTokens
git checkout 273b691d35989d71cd17ff2895fdc735097b92d1
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
python download.py --model
```

Adapt CUDA wheels to the host driver as upstream documents. Install these dependencies in the **SkinTokens environment**, not in the ComfyUI interpreter. Do not assume the Comfy3D image already includes this optional runtime. Download all upstream model assets; copying only the rigging checkpoint is insufficient.

Set these variables in the environment that **starts ComfyUI**, then restart it:

```bash
export SKINTOKENS_ROOT=/opt/SkinTokens
export SKINTOKENS_PYTHON=/opt/SkinTokens/.venv/bin/python
export SKINTOKENS_TIMEOUT_SECONDS=1800  # optional, finite positive seconds
export NO_PROXY=localhost,127.0.0.1
```

Both required environment variables are explicit: there is no fallback to ComfyUI's interpreter. `SKINTOKENS_PYTHON` must be an executable, not a command containing arguments. The checkout must be writable for a job lock and upstream temporary files. All Comfy input/output paths must be visible to that process. Linux is the intended inference platform; the adapter requires POSIX process groups, and Windows is rejected rather than leaking Blender workers.

## Node interface

Required inputs:

- `glb_path` (`STRING`): uploaded relative path under Comfy input, or an absolute GLB path under Comfy input/output/temp. Output paths from previous nodes can be connected directly.
- `output_name` (`STRING`, default `skintokens_rigged.glb`): relative path under Comfy output, optionally including a subfolder.

Optional inputs:

- `use_skeleton` (`BOOLEAN`, default false): use the input's existing skeleton.
- `use_transfer` (`BOOLEAN`, default true): upstream texture/scale transfer path.
- `use_postprocess` (`BOOLEAN`, default false): upstream voxel skin processing.
- `num_beams` (`INT`, default 10, range 1–20).

**Seed is deliberately omitted**: this pinned upstream CLI has no seed argument. The adapter does not silently accept an ignored seed or promise deterministic output. Other sampling parameters retain upstream CLI defaults (top-k 5, top-p 0.95, temperature 1.0, repetition penalty 2.0).

Returns one `STRING` containing the absolute GLB path, plus `ui.files` entries `{filename, subfolder, type: "output"}` for Comfy history/download clients. It is an output node. Example API prompt node:

```json
{
  "class_type": "SkinTokensRigGLB",
  "inputs": {
    "glb_path": "character.glb",
    "output_name": "skintokens/character_rigged.glb",
    "use_skeleton": false,
    "use_transfer": true,
    "use_postprocess": false,
    "num_beams": 10
  }
}
```

## Failure and isolation contract

- Traversal and symlink escapes outside the allowed Comfy directories are rejected. This is path confinement, not a sandbox for malicious local users or malicious model checkpoints.
- **Existing output names are rejected**, never overwritten. Choose a new name for another execution. Every job writes to a new private staging directory, verifies the GLB header/version/declared length, then publishes with an atomic no-clobber link. Missing, empty, malformed, or symlink output cannot reuse a stale destination. Partial staging files are removed after success/failure.
- Nonzero subprocess exit and the last 16 KiB of diagnostics propagate to the node. Timeout terminates the entire worker process group, including Blender and dataloader descendants. The worker reaps its Blender server normally; the outer node also kills leftover group members on exit.
- The worker uses upstream `demo.run_cli`, but owns the Blender lifecycle instead of calling `demo.start_bpy_server` (which creates a separate process group). Before importing upstream code, it reserves a listener on **127.0.0.1:59876** and passes that exact socket to Blender. A private single-request-at-a-time stdlib WSGI adapter replaces Bottle's Tornado backend, retaining the upstream routes and Blender queue. There is no check/close/rebind window in which another checkout can claim the port. `SO_REUSEADDR` permits consecutive jobs after HTTP TIME_WAIT; `SO_REUSEPORT` is not enabled.
- Upstream uses fixed TCP port **59876** and trusted-local pickle messages. Do not expose/forward this port or run untrusted workloads in the same environment. Stop other upstream demos first. Occupied port and concurrent jobs in the same checkout fail rather than sharing a mutable Blender scene. Run one SkinTokens instance per network namespace; separate checkouts do not provide independent server ports.
- The worker enforces loopback bypasses in both `NO_PROXY` and `no_proxy` before upstream imports, so inherited HTTP/ALL proxy settings cannot route the local pickle requests through a remote proxy. This is not a general outbound-network sandbox.
- Comfy's `LD_PRELOAD`, `PYTHONPATH`, and `PYTHONHOME` are removed from child environments. No shell interpolation is used.

## Verification

CPU adapter tests use an explicitly synthetic external `demo.py` fixture and real Python subprocesses. They exercise CLI forwarding, cwd, downloadable result shape, output publication, path confinement, failure diagnostics, occupied ports, and timeout process-tree cleanup. The transport test uses real Bottle/WSGI HTTP GET/POST binary payloads, verifies the Linux listener address, runs consecutive jobs, and checks the port is closed afterward. Proxy tests exercise Requests with conflicting inherited proxy settings. They **do not** validate model inference or anatomical rig quality.

```bash
uv run --no-project --with pytest --with requests --with bottle python -m pytest -q --confcutdir=tests tests
```

`--confcutdir=tests` prevents pytest from importing the Comfy plugin root as a test package. Use a test environment with pytest installed. For actual acceptance, configure the real upstream runtime, execute the node on a GLB, verify Comfy history/download, then import the exact result in isolated Blender and inspect armature, weighted meshes, materials/UVs, posed deformation, and anatomical placement. A valid GLB container alone is not proof of a good rig.

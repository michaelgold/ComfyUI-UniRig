# ComfyUI-UniRig

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (nightly)** — search for `ComfyUI-UniRig` in the Manager and click Install. Fastest, but the Manager's nightly index can lag.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-UniRig.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-UniRig.git
   cd ComfyUI-UniRig
   pip install -r requirements.txt --upgrade
   python install.py
   ```

---



<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-UniRig/">
<img src="https://pozzettiandrea.github.io/ComfyUI-UniRig/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-UniRig/">View Live Test Gallery →</a></b>
</div>

Automatic skeleton extraction for ComfyUI using UniRig (SIGGRAPH 2025) or Make it Animatable (CVPR 2025).
Self-contained with bundled Blender and UniRig/MIA code.

It is recommended to use MIA for humanoid characters.

Rig your character mesh and skin it!
![rigging_and_skinning](docs/rigging_and_skinning.png)

Change their pose, export a new one
![rigging_manipulation](docs/rigging_manipulation.png)



## Video demos

Rigging/skinning workflow (video is sped up for documentation purposes):


https://github.com/user-attachments/assets/6d06a3cd-db63-4e3a-b13b-78ff7868a162


Manipulation/saving/export:


https://github.com/user-attachments/assets/f320db66-4323-4993-a46e-87e2717748ef

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-UniRig/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

- [UniRig Paper](https://zjp-shadow.github.io/works/UniRig/)
- [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)

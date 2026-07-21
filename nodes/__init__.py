"""UniRig Nodes."""

from .mesh_io import UniRigLoadMesh, UniRigSaveMesh
from .load_model import UniRigLoadModel, MIALoadModel
from .auto_rig import UniRigAutoRig
from .skeleton_extraction import UniRigExtractSkeletonNew
from .skinning import UniRigApplySkinningMLNew
from .mia_auto_rig import MIAAutoRig
from .animation import UniRigApplyAnimation
from .skeleton_io import (
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
    UniRigExportPosedFBX,
    UniRigViewRigging,
    UniRigDebugSkeleton,
    UniRigCompareSkeletons,
)
from .orientation_check import UniRigOrientationCheck

NODE_CLASS_MAPPINGS = {
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
    "UniRigLoadModel": UniRigLoadModel,
    "UniRigAutoRig": UniRigAutoRig,
    "UniRigExtractSkeletonNew": UniRigExtractSkeletonNew,
    "UniRigApplySkinningMLNew": UniRigApplySkinningMLNew,
    "MIALoadModel": MIALoadModel,
    "MIAAutoRig": MIAAutoRig,
    "UniRigApplyAnimation": UniRigApplyAnimation,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigExportPosedFBX": UniRigExportPosedFBX,
    "UniRigViewRigging": UniRigViewRigging,
    "UniRigDebugSkeleton": UniRigDebugSkeleton,
    "UniRigCompareSkeletons": UniRigCompareSkeletons,
    "UniRigOrientationCheck": UniRigOrientationCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
    "UniRigLoadModel": "UniRig: (Down)Load Model",
    "UniRigAutoRig": "UniRig: Auto Rig",
    "UniRigExtractSkeletonNew": "UniRig: Extract Skeleton",
    "UniRigApplySkinningMLNew": "UniRig: Apply Skinning ML",
    "MIALoadModel": "MIA: (Down)Load Model",
    "MIAAutoRig": "MIA: Auto Rig",
    "UniRigApplyAnimation": "UniRig: Apply Animation",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigExportPosedFBX": "UniRig: Export Posed FBX",
    "UniRigViewRigging": "UniRig: View Rigging",
    "UniRigDebugSkeleton": "UniRig: Debug Skeleton",
    "UniRigCompareSkeletons": "UniRig: Compare Skeletons",
    "UniRigOrientationCheck": "UniRig: Orientation Check",
}

"""Mesh simplification policy for MIA inference."""

from __future__ import annotations

import inspect
from typing import Any


SIMPLIFICATION_TOLERANCE_PERCENT = 5


def should_simplify_mesh(
    face_count: int,
    target_face_count: int | None,
    *,
    tolerance_percent: int = SIMPLIFICATION_TOLERANCE_PERCENT,
) -> bool:
    """Return whether a reduction is large enough to justify decimation.

    Trimesh decimation does not preserve the textured visual used by MIA's
    export path. Avoid that destructive path when the requested reduction is
    5% or less; the negligible size win is not worth dropping UVs/materials.
    """

    if target_face_count is None or target_face_count <= 0:
        return False
    removed_faces = face_count - target_face_count
    return removed_faces * 100 > face_count * tolerance_percent


def simplify_mesh_if_needed(
    mesh: Any,
    target_face_count: int | None,
    *,
    logger: Any | None = None,
) -> Any:
    """Simplify materially oversized meshes and otherwise preserve them."""

    face_count = len(mesh.faces)
    if not should_simplify_mesh(face_count, target_face_count):
        if (
            logger is not None
            and target_face_count is not None
            and target_face_count > 0
            and face_count > target_face_count
        ):
            logger.info(
                "Skipping MIA mesh simplification from %d to %d faces "
                "because the reduction is within the %.0f%% texture-preservation tolerance",
                face_count,
                target_face_count,
                SIMPLIFICATION_TOLERANCE_PERCENT,
            )
        return mesh

    assert target_face_count is not None
    if logger is not None:
        logger.info(
            "Simplifying MIA input mesh from %d faces to target %d faces...",
            face_count,
            target_face_count,
        )

    simplify = getattr(mesh, "simplify_quadric_decimation", None)
    if simplify is None:
        simplify = getattr(mesh, "simplify_quadratic_decimation", None)
    if simplify is None:
        if logger is not None:
            logger.warning(
                "trimesh simplification is unavailable; continuing with %d faces",
                face_count,
            )
        return mesh

    parameters = inspect.signature(simplify).parameters.values()
    supports_face_count_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "face_count"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )
    if supports_face_count_keyword:
        simplified = simplify(face_count=int(target_face_count))
    else:
        simplified = simplify(int(target_face_count))

    if simplified is not None and hasattr(simplified, "faces") and len(simplified.faces) > 0:
        if logger is not None:
            logger.info(
                "Simplified MIA input mesh to %d vertices, %d faces",
                len(simplified.vertices),
                len(simplified.faces),
            )
        return simplified

    if logger is not None:
        logger.warning(
            "trimesh simplification returned no usable mesh; continuing with %d faces",
            face_count,
        )
    return mesh

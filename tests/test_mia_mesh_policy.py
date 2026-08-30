import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "mia_mesh_policy.py"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location("mia_mesh_policy", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_negligible_reduction_does_not_invoke_texture_destructive_simplification():
    policy = _load_policy_module()

    assert policy.should_simplify_mesh(80_421, 80_000) is False
    assert policy.should_simplify_mesh(84_000, 80_000) is False
    assert policy.should_simplify_mesh(84_210, 80_000) is False


def test_materially_oversized_mesh_is_still_simplified():
    policy = _load_policy_module()

    assert policy.should_simplify_mesh(84_211, 80_000) is True
    assert policy.should_simplify_mesh(100_000, 80_000) is True


def test_disabled_or_nonpositive_targets_never_simplify():
    policy = _load_policy_module()

    assert policy.should_simplify_mesh(100_000, None) is False
    assert policy.should_simplify_mesh(100_000, 0) is False
    assert policy.should_simplify_mesh(100_000, -1) is False


def test_mesh_simplifier_is_only_called_for_material_reductions():
    policy = _load_policy_module()

    class Mesh:
        def __init__(self, face_count):
            self.faces = [None] * face_count
            self.vertices = [None]
            self.simplify_calls = []

        def simplify_quadric_decimation(self, *, face_count):
            self.simplify_calls.append(face_count)
            return Mesh(face_count)

    near_target = Mesh(80_421)
    assert policy.simplify_mesh_if_needed(near_target, 80_000) is near_target
    assert near_target.simplify_calls == []

    oversized = Mesh(100_000)
    simplified = policy.simplify_mesh_if_needed(oversized, 80_000)
    assert oversized.simplify_calls == [80_000]
    assert len(simplified.faces) == 80_000


def test_legacy_positional_simplifier_api_is_supported():
    policy = _load_policy_module()

    class LegacyMesh:
        def __init__(self):
            self.faces = [None] * 100_000
            self.vertices = [None]
            self.calls = []

        def simplify_quadratic_decimation(self, face_count, /):
            self.calls.append(face_count)
            return MeshResult(face_count)

    class MeshResult:
        def __init__(self, face_count):
            self.faces = [None] * face_count
            self.vertices = [None]

    mesh = LegacyMesh()
    simplified = policy.simplify_mesh_if_needed(mesh, 80_000)

    assert len(simplified.faces) == 80_000
    assert mesh.calls == [80_000]


def test_internal_type_error_is_not_retried_positionally():
    policy = _load_policy_module()

    class FailingMesh:
        def __init__(self):
            self.faces = [None] * 100_000
            self.vertices = [None]
            self.calls = []

        def simplify_quadric_decimation(self, *, face_count):
            self.calls.append(face_count)
            raise TypeError("internal simplifier failure")

    mesh = FailingMesh()

    try:
        policy.simplify_mesh_if_needed(mesh, 80_000)
    except TypeError as exc:
        assert str(exc) == "internal simplifier failure"
    else:
        raise AssertionError("internal TypeError should propagate")
    assert mesh.calls == [80_000]


def test_unavailable_or_unusable_simplifier_preserves_original_mesh():
    policy = _load_policy_module()

    class NoSimplifier:
        faces = [None] * 100_000
        vertices = [None]

    class NoneSimplifier(NoSimplifier):
        def simplify_quadric_decimation(self, *, face_count):
            return None

    class EmptySimplifier(NoSimplifier):
        def simplify_quadric_decimation(self, *, face_count):
            result = NoSimplifier()
            result.faces = []
            return result

    for mesh in (NoSimplifier(), NoneSimplifier(), EmptySimplifier()):
        assert policy.simplify_mesh_if_needed(mesh, 80_000) is mesh

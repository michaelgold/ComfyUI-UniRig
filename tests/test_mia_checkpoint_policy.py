import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "nodes" / "mia" / "checkpoint_policy.py"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location("mia_checkpoint_policy", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorLike:
    def __init__(self, shape):
        self.shape = shape


class LazyLinear:
    in_features = 512
    out_features = 52
    weight = None
    bias = None


def test_matching_checkpoint_weight_is_preserved_for_lazy_linear():
    policy = _load_policy_module()
    checkpoint_weight = TensorLike((52, 512))

    adapted = policy.adapt_checkpoint_parameter(
        key="bw_head.weight",
        checkpoint_parameter=checkpoint_weight,
        model_parameter=LazyLinear.weight,
        module=LazyLinear(),
        parameter_name="weight",
    )

    assert adapted is checkpoint_weight


def test_initializer_is_skipped_for_unmaterialized_parameter():
    policy = _load_policy_module()
    initialized = []

    result = policy.initialize_parameter_if_materialized(
        None,
        lambda parameter: initialized.append(parameter),
    )

    assert result is None
    assert initialized == []


if __name__ == "__main__":
    test_matching_checkpoint_weight_is_preserved_for_lazy_linear()
    test_initializer_is_skipped_for_unmaterialized_parameter()

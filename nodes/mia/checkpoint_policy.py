"""Checkpoint adaptation helpers that support lazily initialized layers."""

from __future__ import annotations

from typing import Any, Callable


def initialize_parameter_if_materialized(
    parameter: Any,
    initializer: Callable[[Any], Any],
) -> Any:
    """Initialize an eager parameter and leave AIMDO lazy ``None`` untouched."""
    if parameter is None:
        return None
    return initializer(parameter)


def _declared_parameter_shape(module: Any, parameter_name: str) -> tuple[int, ...]:
    """Return a Linear-compatible parameter shape without materializing it."""
    if parameter_name == "weight" and hasattr(module, "in_features") and hasattr(module, "out_features"):
        return (int(module.out_features), int(module.in_features))
    if parameter_name == "bias" and hasattr(module, "out_features"):
        return (int(module.out_features),)
    raise ValueError(
        f"Cannot determine expected shape for lazy parameter {parameter_name!r} "
        f"on {type(module).__name__}"
    )


def adapt_checkpoint_parameter(
    *,
    key: str,
    checkpoint_parameter: Any,
    model_parameter: Any,
    module: Any,
    parameter_name: str,
) -> Any:
    """Adapt one checkpoint parameter without dereferencing a lazy ``None`` value."""
    checkpoint_shape = tuple(checkpoint_parameter.shape)
    model_shape = (
        _declared_parameter_shape(module, parameter_name)
        if model_parameter is None
        else tuple(model_parameter.shape)
    )
    if checkpoint_shape == model_shape:
        return checkpoint_parameter
    if model_parameter is None:
        raise ValueError(
            f"Checkpoint parameter {key} has shape {checkpoint_shape}, "
            f"expected {model_shape} for the lazy model parameter"
        )
    return model_parameter.to(checkpoint_parameter)

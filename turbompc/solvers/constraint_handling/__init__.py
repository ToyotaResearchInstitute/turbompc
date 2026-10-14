"""Static inequality-handling configuration shared by solver implementations."""

import enum
import math
from typing import Any, Mapping


class ConstraintHandling(enum.IntEnum):
    PROJECTION = 0
    LOG_BARRIER = 1


_CONSTRAINT_HANDLING_BY_NAME = {
    "projection": ConstraintHandling.PROJECTION,
    "log_barrier": ConstraintHandling.LOG_BARRIER,
}


def _validate_constraint_settings(
    params: Mapping[str, Any],
    use_slack_variables: bool,
    slack_penalization_weight: Any,
) -> ConstraintHandling:
    """Validate top-level inequality settings and return the static method."""
    name = params["constraint_handling"]
    if not isinstance(name, str):
        raise TypeError(f"constraint_handling must be a string, got {type(name)}")
    try:
        handling = _CONSTRAINT_HANDLING_BY_NAME[name.lower()]
    except KeyError as exc:
        choices = tuple(_CONSTRAINT_HANDLING_BY_NAME)
        raise ValueError(
            f"Unknown constraint_handling: {name!r}. Options: {choices}"
        ) from exc

    if not isinstance(use_slack_variables, bool):
        raise TypeError("use_slack_variables must be a bool")

    log_barrier_kappa_value = params["log_barrier_kappa"]
    slack_weight_value = slack_penalization_weight
    if getattr(log_barrier_kappa_value, "ndim", 0) != 0:
        raise ValueError("log_barrier_kappa must be a scalar")
    if getattr(slack_weight_value, "ndim", 0) != 0:
        raise ValueError("slack_penalization_weight must be a scalar")
    log_barrier_kappa = float(log_barrier_kappa_value)
    slack_weight = float(slack_weight_value)
    if not math.isfinite(log_barrier_kappa):
        raise ValueError("log_barrier_kappa must be finite")
    if not math.isfinite(slack_weight):
        raise ValueError("slack_penalization_weight must be finite")

    if handling == ConstraintHandling.LOG_BARRIER:
        if log_barrier_kappa <= 0.0:
            raise ValueError(
                "log_barrier_kappa must be positive for log_barrier handling"
            )
    elif log_barrier_kappa != 0.0:
        raise ValueError("log_barrier_kappa must be zero for projection handling")

    if use_slack_variables:
        if slack_weight <= 0.0:
            raise ValueError(
                "slack_penalization_weight must be positive when "
                "use_slack_variables is true"
            )
    elif slack_weight != 0.0:
        raise ValueError(
            "slack_penalization_weight must be zero when use_slack_variables is false"
        )
    return handling


__all__ = ["ConstraintHandling"]

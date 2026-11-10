"""Experiment entry points and reusable runners.

Modules:
- :mod:`experiments.e1_e2`: E1 decomposition quality & E2 routing accuracy (Person A).
- :mod:`experiments.e4`: E4 pipeline control & component ablation (Person B).
"""
from __future__ import annotations

from .e4 import (
    ConditionSpec,
    build_default_config,
    build_expert_registry,
    evaluate_colorization_metrics,
    get_e4_conditions,
    run_pipeline_for_image_ablations,
)

__all__ = [
    "ConditionSpec",
    "build_default_config",
    "build_expert_registry",
    "evaluate_colorization_metrics",
    "get_e4_conditions",
    "run_pipeline_for_image_ablations",
]

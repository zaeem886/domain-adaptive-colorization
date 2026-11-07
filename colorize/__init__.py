"""Region decomposition and routing stages of the MoE colorization pipeline."""
from .config import DecompConfig, DOMAINS, FALLBACK
from .regions import Region, Decomposition, save_decomposition
from .routing import route, resolve_overlaps, label_to_domain, build_label_table

__all__ = [
    "DecompConfig", "DOMAINS", "FALLBACK",
    "Region", "Decomposition", "save_decomposition",
    "route", "resolve_overlaps", "label_to_domain", "build_label_table",
]

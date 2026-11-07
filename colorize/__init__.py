"""Region decomposition and routing stages of the MoE colorization pipeline."""
from .config import DecompConfig, DOMAINS, FALLBACK, DEFAULT_PROMPTS
from .regions import Region, Decomposition, save_decomposition
from .routing import route, resolve_overlaps, label_to_domain, build_label_table
from .filtering import filter_detections, merge_same_label
from .decomposition import load_gray   # PIL only; torch is imported lazily by GroundedSAM

__all__ = [
    "DecompConfig", "DOMAINS", "FALLBACK", "DEFAULT_PROMPTS",
    "Region", "Decomposition", "save_decomposition",
    "route", "resolve_overlaps", "label_to_domain", "build_label_table",
    "filter_detections", "merge_same_label", "load_gray",
]

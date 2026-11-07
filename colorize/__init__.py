"""Region decomposition, routing, and colorization pipeline for MoE image colorization."""
from .config import DecompConfig, DOMAINS, FALLBACK, DEFAULT_PROMPTS
from .regions import Region, Decomposition, save_decomposition
from .routing import route, resolve_overlaps, label_to_domain, build_label_table
from .filtering import filter_detections, merge_same_label
from .decomposition import load_gray   # PIL only; torch is imported lazily by GroundedSAM

# --- Colorization pipeline (Person B) ---
from .colorization_config import ColorizationConfig
from .extraction import RegionExtractor, CropBox
from .blending import Blender, Reconstructor
from .experts import ExpertManager, ExpertRegistry, Colorizer, default_registry
from .pipeline import ColorizationPipeline

__all__ = [
    # decomposition / routing
    "DecompConfig", "DOMAINS", "FALLBACK", "DEFAULT_PROMPTS",
    "Region", "Decomposition", "save_decomposition",
    "route", "resolve_overlaps", "label_to_domain", "build_label_table",
    "filter_detections", "merge_same_label", "load_gray",
    # colorization pipeline
    "ColorizationConfig",
    "RegionExtractor", "CropBox",
    "Blender", "Reconstructor",
    "ExpertManager", "ExpertRegistry", "Colorizer", "default_registry",
    "ColorizationPipeline",
]

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

# --- Metrics (Person C) ---
from .metrics import (
    compute_psnr, compute_ssim, compute_lab_delta_e, compute_boundary_delta_e,
    compute_colorfulness, compute_lpips, compute_fid,
    compute_miou, compute_pixel_coverage, compute_routing_accuracy, compute_confusion_matrix,
)
from .results import ResultsRecord, save_results, load_results, append_result, records_to_dataframe
from .test_sets import (
    DOMAIN_SET_IDS, MIXED_SCENE_IDS, SEGMENTATION_SET_IDS, REGION_CROP_IDS,
    ALL_TEST_IDS, COCO_DOMAIN_CATEGORIES, get_coco_image_path, load_test_set,
    find_duplicate_ids, validate_coco_ids, write_coco_metadata_manifest,
)
from .plotting import (
    plot_metric_bar, plot_metric_comparison, plot_confusion_matrix,
    plot_colorfulness_distribution, plot_delta_e_map,
)

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
    # --- Metrics (Person C) ---
    "compute_psnr", "compute_ssim", "compute_lab_delta_e", "compute_boundary_delta_e",
    "compute_colorfulness", "compute_lpips", "compute_fid",
    "compute_miou", "compute_pixel_coverage", "compute_routing_accuracy", "compute_confusion_matrix",
    "ResultsRecord", "save_results", "load_results", "append_result", "records_to_dataframe",
    "DOMAIN_SET_IDS", "MIXED_SCENE_IDS", "SEGMENTATION_SET_IDS", "REGION_CROP_IDS",
    "ALL_TEST_IDS", "COCO_DOMAIN_CATEGORIES", "get_coco_image_path", "load_test_set",
    "find_duplicate_ids", "validate_coco_ids", "write_coco_metadata_manifest",
    "plot_metric_bar", "plot_metric_comparison", "plot_confusion_matrix",
    "plot_colorfulness_distribution", "plot_delta_e_map",
]

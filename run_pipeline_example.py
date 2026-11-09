#!/usr/bin/env python3
"""Example end-to-end execution script for the MoE Image Colorization Pipeline.

Demonstrates:
  Step 1: Load grayscale input (L channel)
  Step 2: Person A (GroundedSAM / Routing -> Decomposition)
  Step 3: Person B (Region Extraction -> Expert Colorization -> Soft Blending -> RGB Reconstruction)
"""
import numpy as np
from PIL import Image

from colorize.config import DecompConfig
from colorize.colorization_config import ColorizationConfig
from colorize.decomposition import load_gray, GroundedSAM
from colorize.pipeline import ColorizationPipeline
from colorize.experts import ExpertRegistry


def run_pipeline(
    image_path: str,
    checkpoints: ExpertRegistry,
    margin: float = 0.2,
    smoothing_width: float = 10.0,
    blending_on: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """Run full pipeline on an input image."""
    # 1. Load image and extract L channel
    img_gray = load_gray(image_path, max_side=1024)
    orig_pil = Image.open(image_path)
    original_shape = (orig_pil.height, orig_pil.width)

    # Convert to float32 L channel [0, 100]
    L_full = np.array(img_gray, dtype=np.float32) * (100.0 / 255.0)

    # 2. Person A: Grounded-SAM Detection & Routing
    decomp_cfg = DecompConfig()
    grounded_sam = GroundedSAM(cfg=decomp_cfg, device=device)
    decomposition = grounded_sam(img_gray)

    print(f"[Person A] Routed {len(decomposition.regions)} regions.")
    print(f"[Person A] Domain coverage: {decomposition.coverage * 100:.1f}%")
    print(f"[Person A] Domain breakdown: {decomposition.domain_share}")

    # 3. Person B: Colorization Pipeline (Extraction, Expert routing, Blending, Reconstruction)
    color_cfg = ColorizationConfig(
        margin=margin,
        smoothing_width=smoothing_width,
        blending_on=blending_on,
        region_shape="mask",
        input_size=256,
    )

    pipeline = ColorizationPipeline(registry=checkpoints, cfg=color_cfg, device=device)
    rgb_output = pipeline.run(
        L_full=L_full,
        decomp=decomposition,
        original_shape=original_shape,
    )

    print(f"[Person B] Colorization complete. Output RGB shape: {rgb_output.shape}")
    return rgb_output


if __name__ == "__main__":
    print("Example pipeline usage module created.")
    print("Call run_pipeline(image_path, checkpoints, ...) with your model checkpoints.")

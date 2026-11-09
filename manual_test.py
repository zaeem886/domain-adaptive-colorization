#!/usr/bin/env python3
"""Interactive manual test script for Person A + Person B pipeline.

Takes a real image, creates regions (either simulated distinct objects or from routing),
runs contextual cropping, simulates domain experts with distinct colors,
blends them using the Gaussian soft masks, and writes out visual artifacts:
  - manual_test_output/01_input_gray.png
  - manual_test_output/02_person_a_domain_map.png
  - manual_test_output/03_blending_on_colorized.png
  - manual_test_output/04_blending_off_hard_paste.png
  - manual_test_output/05_comparison_side_by_side.png
"""
from __future__ import annotations

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from colorize.config import DecompConfig
from colorize.regions import Region
from colorize.routing import route
from colorize.colorization_config import ColorizationConfig
from colorize.extraction import RegionExtractor
from colorize.blending import Blender, Reconstructor


def create_color_palette():
    """Distinct visible chromaticities for each domain in CIE-Lab space:
    L is from image, ab provides color:
      - 'nature'   : lush green (a ≈ -45, b ≈ 45)
      - 'people'   : warm skin tone (a ≈ 25, b ≈ 25)
      - 'urban'    : brick red / terracotta (a ≈ 40, b ≈ 30)
      - 'animals'  : golden/amber fur (a ≈ 15, b ≈ 50)
      - 'fallback' : cool desaturated blue-gray (a ≈ -5, b ≈ -20)
    """
    return {
        "nature": np.array([-45.0, 45.0], dtype=np.float32),
        "people": np.array([25.0, 25.0], dtype=np.float32),
        "urban": np.array([40.0, 30.0], dtype=np.float32),
        "animals": np.array([15.0, 50.0], dtype=np.float32),
        "fallback": np.array([-5.0, -20.0], dtype=np.float32),
    }


def run_manual_test(image_path: str, output_dir: str = "manual_test_output"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Manual Pipeline Testing on: {image_path} ===")

    # 1. Load input image & convert to grayscale (L channel)
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize to manageable working size for demonstration (e.g. max side 600)
    scale = 600.0 / max(orig_w, orig_h) if max(orig_w, orig_h) > 600 else 1.0
    work_w, work_h = int(orig_w * scale), int(orig_h * scale)
    pil_work = pil_img.resize((work_w, work_h), Image.Resampling.LANCZOS)
    gray_work = pil_work.convert("L")

    # Save input grayscale
    gray_work.save(os.path.join(output_dir, "01_input_gray.png"))
    print(f"[Step 1] Loaded image ({orig_w}x{orig_h}), working resolution: ({work_w}x{work_h})")

    # Convert grayscale [0, 255] to Lab L channel [0, 100]
    L_full = np.array(gray_work, dtype=np.float32) * (100.0 / 255.0)

    # 2. Person A Simulation: Create 3 meaningful geometric regions
    # Region 1: "sky" / nature in top portion
    mask_nature = np.zeros((work_h, work_w), dtype=bool)
    mask_nature[0 : int(work_h * 0.45), :] = True
    r_nature = Region(
        label="sky",
        score=0.95,
        box=np.array([0, 0, work_w, int(work_h * 0.45)], dtype=float),
        mask=mask_nature
    )

    # Region 2: "person" / people in lower-left
    mask_people = np.zeros((work_h, work_w), dtype=bool)
    p_x0, p_x1 = int(work_w * 0.1), int(work_w * 0.45)
    p_y0, p_y1 = int(work_h * 0.40), int(work_h * 0.85)
    mask_people[p_y0:p_y1, p_x0:p_x1] = True
    r_people = Region(
        label="person",
        score=0.90,
        box=np.array([p_x0, p_y0, p_x1, p_y1], dtype=float),
        mask=mask_people
    )

    # Region 3: "building" / urban in lower-right
    mask_urban = np.zeros((work_h, work_w), dtype=bool)
    u_x0, u_x1 = int(work_w * 0.50), int(work_w * 0.90)
    u_y0, u_y1 = int(work_h * 0.45), int(work_h * 0.90)
    mask_urban[u_y0:u_y1, u_x0:u_x1] = True
    r_urban = Region(
        label="building",
        score=0.85,
        box=np.array([u_x0, u_y0, u_x1, u_y1], dtype=float),
        mask=mask_urban
    )

    # Run Person A's Routing module
    decomp_cfg = DecompConfig()
    decomposition = route([r_nature, r_people, r_urban], (work_h, work_w), decomp_cfg)

    print(f"[Step 2 - Person A] Routing completed:")
    print(f"  - Active regions: {[r.label + ' -> ' + r.domain for r in decomposition.regions]}")
    print(f"  - Pixel coverage: {decomposition.coverage * 100:.2f}%")
    print(f"  - Domain shares:  {decomposition.domain_share}")

    # Visualize Person A's domain map as a color-coded inspection image
    domain_colors = {
        "nature": [34, 139, 34],       # Forest green
        "people": [255, 140, 0],       # Orange
        "urban": [70, 130, 180],       # Steel blue
        "animals": [160, 82, 45],      # Sienna
        "fallback": [128, 128, 128],   # Gray
    }
    domain_vis = np.zeros((work_h, work_w, 3), dtype=np.uint8)
    for d_idx, d_name in enumerate(decomposition.domains):
        domain_vis[decomposition.domain_map == d_idx] = domain_colors.get(d_name, [0, 0, 0])
    Image.fromarray(domain_vis).save(os.path.join(output_dir, "02_person_a_domain_map.png"))

    # 3. Person B Colorization Pipeline: Test both Soft Blending ON and OFF
    palette = create_color_palette()

    for blending_active, mode_name in [(True, "03_blending_on_colorized.png"), (False, "04_blending_off_hard_paste.png")]:
        color_cfg = ColorizationConfig(
            margin=0.25,
            smoothing_width=15.0,
            blending_on=blending_active,
            region_shape="mask",
            input_size=256,
        )

        extractor = RegionExtractor(color_cfg, (work_h, work_w))
        blender = Blender(color_cfg, (work_h, work_w))

        # Build fallback region for uncovered areas
        fallback_mask = decomposition.owner < 0
        ys, xs = np.where(fallback_mask)
        fb_box = (
            np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
            if len(ys) > 0
            else np.array([0, 0, work_w, work_h], dtype=float)
        )
        fallback_region = Region(
            label="fallback",
            score=1.0,
            box=fb_box,
            mask=fallback_mask,
            domain="fallback",
        )

        all_regions = decomposition.regions + [fallback_region]
        ab_canvases, binary_masks, crops = [], [], []

        for reg in all_regions:
            L_patch, crop = extractor.extract_input(L_full, reg)
            assert L_patch.shape == (256, 256)

            # Expert colorizer emulation (predicts domain chrominance)
            dom_color = palette.get(reg.domain, np.array([0.0, 0.0], dtype=np.float32))
            ab_patch = np.zeros((256, 256, 2), dtype=np.float32)
            ab_patch[:, :] = dom_color

            # Place back onto full-image canvas
            canvas = extractor.map_output(ab_patch, crop)
            mask_b = extractor.get_binary_mask(reg, crop)

            ab_canvases.append(canvas)
            binary_masks.append(mask_b)
            crops.append(crop)

        # Blend chrominance
        ab_blended = blender.blend(ab_canvases, binary_masks, crops)

        # Reconstruct to RGB
        reconstructor = Reconstructor(original_shape=(work_h, work_w), working_shape=(work_h, work_w))
        rgb_res = reconstructor.reconstruct(L_full, ab_blended)

        out_path = os.path.join(output_dir, mode_name)
        Image.fromarray(rgb_res).save(out_path)
        print(f"[Step 3 - Person B] Saved: {out_path} (blending_on={blending_active})")

    # 4. Generate side-by-side comparison image
    img_gray_rgb = Image.open(os.path.join(output_dir, "01_input_gray.png")).convert("RGB")
    img_map = Image.open(os.path.join(output_dir, "02_person_a_domain_map.png"))
    img_blend_off = Image.open(os.path.join(output_dir, "04_blending_off_hard_paste.png"))
    img_blend_on = Image.open(os.path.join(output_dir, "03_blending_on_colorized.png"))

    total_w = work_w * 4
    comparison = Image.new("RGB", (total_w, work_h))
    comparison.paste(img_gray_rgb, (0, 0))
    comparison.paste(img_map, (work_w, 0))
    comparison.paste(img_blend_off, (work_w * 2, 0))
    comparison.paste(img_blend_on, (work_w * 3, 0))

    comp_path = os.path.join(output_dir, "05_comparison_side_by_side.png")
    comparison.save(comp_path)
    print(f"\n[Finished] Side-by-side comparison created at: {comp_path}")
    print(f"Panels: [1] Input Grayscale | [2] Person A Domain Map | [3] Hard Paste (Blend OFF) | [4] Soft Blended (Blend ON)")


if __name__ == "__main__":
    test_img = "pics/2. DDColor architecture — Section 3.2.2.5.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    run_manual_test(test_img)

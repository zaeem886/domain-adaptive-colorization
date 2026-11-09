#!/usr/bin/env python3
"""Integration smoke test for Person A (decomposition & routing) + Person B (colorization).

Validates the complete handover contract without a GPU or model weights:
1. Synthetic luminance image.
2. Person A: synthetic regions routed via route().
3. Person B: contextual crops, mocked per-domain experts, soft blending,
   and full-resolution RGB reconstruction.
"""
from __future__ import annotations

import numpy as np

from colorize.blending import Blender, Reconstructor
from colorize.colorization_config import ColorizationConfig
from colorize.config import DecompConfig
from colorize.extraction import RegionExtractor
from colorize.regions import Region
from colorize.routing import route


class MockExpertColorizer:
    """Mock expert that predicts constant distinctive chrominance per domain."""

    DOMAIN_COLORS = {
        "nature": np.array([-20.0, 30.0], dtype=np.float32),   # greenish
        "people": np.array([25.0, 15.0], dtype=np.float32),    # skin-like warm
        "urban": np.array([5.0, -10.0], dtype=np.float32),     # cool bluish
        "fallback": np.array([0.0, 0.0], dtype=np.float32),    # neutral
    }

    def __init__(self, domain: str):
        self.domain = domain

    def predict(self, L: np.ndarray) -> np.ndarray:
        S = L.shape[0]
        color = self.DOMAIN_COLORS.get(self.domain, np.array([0.0, 0.0], dtype=np.float32))
        ab = np.zeros((S, S, 2), dtype=np.float32)
        ab[:, :] = color
        return ab


def test_person_a_and_b_integration():
    # 1. Setup image (luminance channel)
    H, W = 120, 160
    rng = np.random.default_rng(42)
    L_full = rng.uniform(20.0, 80.0, (H, W)).astype(np.float32)

    # 2. Person A: detections -> region decomposition -> routing
    mask_nature = np.zeros((H, W), dtype=bool)
    mask_nature[0:50, 0:W] = True
    r_nature = Region(
        label="sky",
        score=0.92,
        box=np.array([0, 0, W, 50], dtype=float),
        mask=mask_nature,
    )

    mask_person = np.zeros((H, W), dtype=bool)
    mask_person[50:110, 10:70] = True
    r_person = Region(
        label="person",
        score=0.88,
        box=np.array([10, 50, 70, 110], dtype=float),
        mask=mask_person,
    )

    decomposition = route([r_nature, r_person], (H, W), DecompConfig())

    # Verify Person A output contracts
    assert len(decomposition.regions) == 2
    assert decomposition.regions[0].domain == "nature"
    assert decomposition.regions[1].domain == "people"
    assert "fallback" in decomposition.domains
    assert (decomposition.owner >= -1).all()

    # 3. Person B: colorization execution
    color_cfg = ColorizationConfig(margin=0.2, smoothing_width=8.0, blending_on=True)
    extractor = RegionExtractor(color_cfg, (H, W))
    blender = Blender(color_cfg, (H, W))

    # Build fallback region (uncovered pixels)
    fallback_mask = decomposition.owner < 0
    assert fallback_mask.any(), "Expected some pixels in fallback"

    ys, xs = np.where(fallback_mask)
    fallback_box = np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float
    )
    fallback_region = Region(
        label="fallback",
        score=1.0,
        box=fallback_box,
        mask=fallback_mask,
        domain="fallback",
    )

    all_regions = decomposition.regions + [fallback_region]

    # Extract, colorize with mock experts, map to canvas
    ab_canvases = []
    binary_masks = []
    crops = []

    for region in all_regions:
        L_patch, crop = extractor.extract_input(L_full, region)
        assert L_patch.shape == (color_cfg.input_size, color_cfg.input_size)

        expert = MockExpertColorizer(region.domain)
        ab_patch = expert.predict(L_patch)

        canvas = extractor.map_output(ab_patch, crop)
        mask_b = extractor.get_binary_mask(region, crop)

        ab_canvases.append(canvas)
        binary_masks.append(mask_b)
        crops.append(crop)

    # 4. Blend chrominance
    ab_blended = blender.blend(ab_canvases, binary_masks, crops)

    assert ab_blended.shape == (H, W, 2)
    assert np.isfinite(ab_blended).all(), "Found NaN/Inf in blended chrominance"

    # In the pure nature region, chrominance should be close to the nature color
    nature_center = ab_blended[20, 80]
    np.testing.assert_allclose(
        nature_center, MockExpertColorizer.DOMAIN_COLORS["nature"], atol=1.0
    )

    # 5. Full reconstruction to RGB
    reconstructor = Reconstructor(original_shape=(H, W), working_shape=(H, W))
    rgb = reconstructor.reconstruct(L_full, ab_blended)

    assert rgb.shape == (H, W, 3)
    assert rgb.dtype == np.uint8
    assert (rgb >= 0).all() and (rgb <= 255).all()

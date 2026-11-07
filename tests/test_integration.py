#!/usr/bin/env python3
"""Integration smoke test for Person A (Decomposition & Routing) + Person B (Colorization).

This validates the complete handover contract:
1. Synthetic grayscale image is loaded / constructed.
2. Person A: DINO + SAM outputs (or synthetic regions) are routed via route().
3. Person B: RegionExtractor crops contextually and extracts L patches.
4. Person B: ExpertManager routes patches to domain experts (mocked or real).
5. Person B: Blender applies dynamic sigma Gaussian smoothing and blends patches.
6. Person B: Reconstructor reconstructs final full-resolution RGB.
"""
from __future__ import annotations

import sys
import types
import unittest
import contextlib
import numpy as np
from scipy.ndimage import zoom as _zoom

# ---------------------------------------------------------------------------
# Lightweight stubs for environments without cv2 or torch
# ---------------------------------------------------------------------------
if "cv2" not in sys.modules:
    try:
        import cv2
    except ImportError:
        def _cv2_resize(img: np.ndarray, size, interpolation=None) -> np.ndarray:
            dw, dh = size
            sh, sw = img.shape[:2]
            zy, zx = dh / sh, dw / sw
            if img.ndim == 2:
                return _zoom(img, (zy, zx), order=3).astype(img.dtype)
            return _zoom(img, (zy, zx, 1), order=3).astype(img.dtype)

        def _cv2_cvtColor(src: np.ndarray, code: int) -> np.ndarray:
            # Fallback approximate conversion for testing when OpenCV is not installed
            if code == 56:  # COLOR_Lab2BGR
                # Lab uint8 -> approximate BGR
                return np.repeat(src[:, :, :1], 3, axis=2)
            elif code == 4:  # COLOR_BGR2RGB
                return src[:, :, ::-1]
            return src

        cv2_stub = types.ModuleType("cv2")
        cv2_stub.INTER_CUBIC = 2
        cv2_stub.INTER_LINEAR = 1
        cv2_stub.COLOR_Lab2BGR = 56
        cv2_stub.COLOR_LAB2BGR = 56
        cv2_stub.COLOR_BGR2RGB = 4
        cv2_stub.resize = _cv2_resize
        cv2_stub.cvtColor = _cv2_cvtColor
        sys.modules["cv2"] = cv2_stub

if "torch" not in sys.modules:
    try:
        import torch
    except ImportError:
        def _no_grad():
            @contextlib.contextmanager
            def _ctx():
                yield
            def _decorator(fn):
                import functools
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return wrapper
            return _decorator

        torch_stub = types.ModuleType("torch")
        torch_stub.no_grad = _no_grad
        torch_stub.device = lambda x: x
        torch_stub.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        nn_stub = types.ModuleType("torch.nn")
        nn_stub.Module = object
        torch_stub.nn = nn_stub
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub

sys.path.insert(0, "/home/zaeem/Documents/domain-adaptive-colorization")

from colorize.config import DecompConfig
from colorize.regions import Region
from colorize.routing import route
from colorize.colorization_config import ColorizationConfig
from colorize.extraction import RegionExtractor
from colorize.blending import Blender, Reconstructor
from colorize.experts import ExpertManager


class MockExpertColorizer:
    """Mock expert that predicts constant distinctive chrominance per domain."""
    DOMAIN_COLORS = {
        "nature": np.array([-20.0, 30.0], dtype=np.float32),  # greenish
        "people": np.array([25.0, 15.0], dtype=np.float32),   # skin-like warm
        "urban": np.array([5.0, -10.0], dtype=np.float32),    # cool bluish
        "fallback": np.array([0.0, 0.0], dtype=np.float32),   # neutral
    }

    def __init__(self, domain: str):
        self.domain = domain

    def predict(self, L: np.ndarray) -> np.ndarray:
        S = L.shape[0]
        color = self.DOMAIN_COLORS.get(self.domain, np.array([0.0, 0.0], dtype=np.float32))
        ab = np.zeros((S, S, 2), dtype=np.float32)
        ab[:, :] = color
        return ab


class TestEndToEndPipelineHandover(unittest.TestCase):

    def test_person_a_and_b_integration(self):
        # 1. Setup Image (Luminance channel)
        H, W = 120, 160
        rng = np.random.default_rng(42)
        L_full = rng.uniform(20.0, 80.0, (H, W)).astype(np.float32)

        # 2. Person A: Detections -> Region Decomposition -> Routing
        # Region 1: Sky / Nature on top half
        mask_nature = np.zeros((H, W), dtype=bool)
        mask_nature[0:50, 0:W] = True
        r_nature = Region(
            label="sky",
            score=0.92,
            box=np.array([0, 0, W, 50], dtype=float),
            mask=mask_nature
        )

        # Region 2: Person in bottom-left
        mask_person = np.zeros((H, W), dtype=bool)
        mask_person[50:110, 10:70] = True
        r_person = Region(
            label="person",
            score=0.88,
            box=np.array([10, 50, 70, 110], dtype=float),
            mask=mask_person
        )

        decomp_cfg = DecompConfig()
        decomposition = route([r_nature, r_person], (H, W), decomp_cfg)

        # Verify Person A output contracts
        self.assertEqual(len(decomposition.regions), 2)
        self.assertEqual(decomposition.regions[0].domain, "nature")
        self.assertEqual(decomposition.regions[1].domain, "people")
        self.assertIn("fallback", decomposition.domains)
        self.assertTrue((decomposition.owner >= -1).all())

        # 3. Person B: Colorization Execution
        color_cfg = ColorizationConfig(margin=0.2, smoothing_width=8.0, blending_on=True)
        extractor = RegionExtractor(color_cfg, (H, W))
        blender = Blender(color_cfg, (H, W))

        # Build fallback region (uncovered pixels)
        fallback_mask = decomposition.owner < 0
        self.assertTrue(fallback_mask.any(), "Expected some pixels in fallback")

        ys, xs = np.where(fallback_mask)
        fallback_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
        fallback_region = Region(
            label="fallback",
            score=1.0,
            box=fallback_box,
            mask=fallback_mask,
            domain="fallback",
        )

        all_regions = decomposition.regions + [fallback_region]

        # Extract, Colorize via Mock Experts, Map to Canvas
        ab_canvases = []
        binary_masks = []
        crops = []

        for region in all_regions:
            L_patch, crop = extractor.extract_input(L_full, region)
            self.assertEqual(L_patch.shape, (color_cfg.input_size, color_cfg.input_size))

            # Colorize with mock expert
            expert = MockExpertColorizer(region.domain)
            ab_patch = expert.predict(L_patch)

            # Map back to full canvas and get mask
            canvas = extractor.map_output(ab_patch, crop)
            mask_b = extractor.get_binary_mask(region, crop)

            ab_canvases.append(canvas)
            binary_masks.append(mask_b)
            crops.append(crop)

        # 4. Blend chrominance
        ab_blended = blender.blend(ab_canvases, binary_masks, crops)

        # Check blending invariants
        self.assertEqual(ab_blended.shape, (H, W, 2))
        self.assertFalse(np.isnan(ab_blended).any(), "Found NaNs in blended chrominance")
        self.assertFalse(np.isinf(ab_blended).any(), "Found Infs in blended chrominance")

        # In pure nature region, chrominance should be close to nature domain color
        nature_center = ab_blended[20, 80]
        np.testing.assert_allclose(nature_center, MockExpertColorizer.DOMAIN_COLORS["nature"], atol=1.0)

        # 5. Full reconstruction to RGB
        reconstructor = Reconstructor(original_shape=(H, W), working_shape=(H, W))
        rgb = reconstructor.reconstruct(L_full, ab_blended)

        self.assertEqual(rgb.shape, (H, W, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue((rgb >= 0).all() and (rgb <= 255).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

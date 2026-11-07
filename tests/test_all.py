#!/usr/bin/env python3
"""Unified test suite for Person A (Routing & Decomposition) + Person B (Colorization).

Runs all Person A routing tests and Person B colorization unit & integration tests
using Python's built-in `unittest` module (so no pytest or external packages are needed).
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest

import numpy as np
from scipy.ndimage import zoom as _zoom

# ---------------------------------------------------------------------------
# Transparent Fallbacks if cv2 or torch are not installed in current environment
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
            if code == 56:  # COLOR_Lab2BGR
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
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        nn_stub = types.ModuleType("torch.nn")
        nn_stub.Module = object
        torch_stub.nn = nn_stub
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub

sys.path.insert(0, "/home/zaeem/Documents/domain-adaptive-colorization")

from colorize.config import DecompConfig
from colorize.regions import Region
from colorize.routing import route, resolve_overlaps, label_to_domain, build_label_table
from colorize.colorization_config import ColorizationConfig
from colorize.extraction import CropBox, RegionExtractor
from colorize.blending import Blender, Reconstructor


# ===========================================================================
# Person A Tests (Routing, Domain Mapping, Overlap Resolution)
# ===========================================================================

H_A, W_A = 20, 30

def _reg_a(label, y0, y1, x0, x1, score=0.5):
    m = np.zeros((H_A, W_A), dtype=bool)
    m[y0:y1, x0:x1] = True
    return Region(label=label, score=score, box=np.array([x0, y0, x1, y1], float), mask=m)


class TestPersonARouting(unittest.TestCase):

    def setUp(self):
        self.table = build_label_table(DecompConfig().prompts)

    def test_exact_label_maps_to_domain(self):
        self.assertEqual(label_to_domain("face", self.table), "people")
        self.assertEqual(label_to_domain("sky", self.table), "nature")
        self.assertEqual(label_to_domain("building", self.table), "urban")
        self.assertEqual(label_to_domain("person", self.table), "people")

    def test_merged_label_is_split(self):
        self.assertEqual(label_to_domain("person face", self.table, split_merged=True), "people")

    def test_unknown_label_returns_none(self):
        self.assertIsNone(label_to_domain("spaceship", self.table))

    def test_uncovered_pixels_go_to_fallback(self):
        d = route([_reg_a("person", 0, 10, 0, 10)], (H_A, W_A))
        self.assertAlmostEqual(d.coverage, 100 / (H_A * W_A))
        self.assertEqual(d.domain_map[15, 25], d.domains.index("fallback"))
        self.assertEqual(d.domain_map[5, 5], d.domains.index("people"))

    def test_every_pixel_gets_exactly_one_domain(self):
        regs = [_reg_a("person", 0, 12, 0, 12), _reg_a("sky", 5, 20, 5, 30), _reg_a("animal", 8, 14, 8, 14)]
        d = route(regs, (H_A, W_A))
        self.assertGreaterEqual(int(d.domain_map.min()), 0)
        self.assertLess(int(d.domain_map.max()), len(d.domains))

    def test_domain_share_sums_to_one(self):
        d = route([_reg_a("person", 0, 10, 0, 10), _reg_a("sky", 10, 20, 0, 30)], (H_A, W_A))
        self.assertAlmostEqual(sum(d.domain_share.values()), 1.0)

    def test_smaller_region_wins_overlap_by_default(self):
        big = _reg_a("sky", 0, 20, 0, 30, score=0.9)
        small = _reg_a("person", 5, 10, 5, 10, score=0.4)
        d = route([big, small], (H_A, W_A))
        self.assertEqual(d.domain_map[7, 7], d.domains.index("people"))
        self.assertEqual(d.domain_map[15, 25], d.domains.index("nature"))

    def test_resolved_masks_are_disjoint_and_cover_union(self):
        regs = [_reg_a("person", 0, 12, 0, 12), _reg_a("sky", 5, 20, 5, 30)]
        d = route(regs, (H_A, W_A))
        total = sum(d.resolved_mask(i).sum() for i in range(len(regs)))
        union = (regs[0].mask | regs[1].mask).sum()
        self.assertEqual(total, union)


# ===========================================================================
# Person B Tests (Region Extraction, Blending, Reconstruction)
# ===========================================================================

H_B, W_B = 100, 150

def _reg_b(x0, y0, x1, y1, label="sky", domain="nature"):
    mask = np.zeros((H_B, W_B), dtype=bool)
    mask[y0:y1, x0:x1] = True
    box = np.array([x0, y0, x1, y1], dtype=float)
    return Region(label=label, score=0.9, box=box, mask=mask, domain=domain)


class TestPersonBExtraction(unittest.TestCase):

    def test_contextual_margin_expansion(self):
        cfg = ColorizationConfig(margin=0.2)
        ex = RegionExtractor(cfg, (H_B, W_B))
        r = _reg_b(30, 20, 70, 60)  # w=40, h=40 -> 0.2*40 = 8 margin
        crop = ex.get_crop_box(r)
        self.assertEqual((crop.x0, crop.y0, crop.x1, crop.y1), (22, 12, 78, 68))

    def test_boundary_clipping(self):
        cfg = ColorizationConfig(margin=0.5)
        ex = RegionExtractor(cfg, (H_B, W_B))
        r = _reg_b(0, 0, 20, 20)
        crop = ex.get_crop_box(r)
        self.assertGreaterEqual(crop.x0, 0)
        self.assertGreaterEqual(crop.y0, 0)
        self.assertLessEqual(crop.x1, W_B)
        self.assertLessEqual(crop.y1, H_B)

    def test_output_mapping_preserves_zero_outside(self):
        cfg = ColorizationConfig(margin=0.0, input_size=32)
        ex = RegionExtractor(cfg, (H_B, W_B))
        r = _reg_b(40, 30, 80, 70)
        L_dummy = np.zeros((H_B, W_B), dtype=np.float32)
        _, crop = ex.extract_input(L_dummy, r)
        ab_pred = np.full((32, 32, 2), 5.0, dtype=np.float32)
        canvas = ex.map_output(ab_pred, crop)

        outside = np.ones((H_B, W_B), dtype=bool)
        outside[crop.y0:crop.y1, crop.x0:crop.x1] = False
        self.assertAlmostEqual(float(canvas[outside].sum()), 0.0)


class TestPersonBBlending(unittest.TestCase):

    def test_dynamic_sigma_formula_and_cap(self):
        # 1. Formula when below cap: ρ·min(w,h)/3
        cfg = ColorizationConfig(margin=0.2, smoothing_width=100.0)
        blender = Blender(cfg, (H_B, W_B))
        crop = CropBox(10, 10, 40, 30)  # w=30, h=20 -> 0.2*20/3 = 1.333
        self.assertAlmostEqual(blender.sigma_for_region(crop), 0.2 * 20 / 3.0, places=4)

        # 2. Capped by sigma0
        cfg_capped = ColorizationConfig(margin=0.2, smoothing_width=10.0)
        blender_capped = Blender(cfg_capped, (400, 500))
        big_crop = CropBox(0, 0, 500, 300)  # 0.2*300/3 = 20 > 10
        self.assertAlmostEqual(blender_capped.sigma_for_region(big_crop), 10.0)

    def test_division_by_zero_safety(self):
        cfg = ColorizationConfig(blending_on=False)
        blender = Blender(cfg, (H_B, W_B))
        ab = np.full((H_B, W_B, 2), 5.0, dtype=np.float32)
        bm = np.zeros((H_B, W_B), dtype=bool)
        bm[0:50, 0:50] = True
        crop = CropBox(0, 0, 50, 50)
        result = blender.blend([ab], [bm], [crop])
        uncovered = result[50:, 50:]
        self.assertTrue(np.isfinite(uncovered).all())
        np.testing.assert_array_equal(uncovered, 0.0)


# ===========================================================================
# End-to-End Handover (Person A + Person B together)
# ===========================================================================

class TestEndToEndHandover(unittest.TestCase):

    def test_full_pipeline_flow(self):
        from tests.test_integration import TestEndToEndPipelineHandover
        test_case = TestEndToEndPipelineHandover()
        test_case.test_person_a_and_b_integration()


if __name__ == "__main__":
    unittest.main(verbosity=2)

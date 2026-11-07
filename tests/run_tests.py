#!/usr/bin/env python3
"""Standalone test runner (no pytest, no cv2, no torch needed).

Installs thin stubs for cv2 and torch before importing any project code,
then runs all test cases in test_colorization.py using unittest.

Usage:
    python3 run_tests.py [-v]
"""
from __future__ import annotations

import sys
import types
import unittest
import contextlib

import numpy as np
from scipy.ndimage import zoom as _zoom

# ---------------------------------------------------------------------------
# cv2 stub
# ---------------------------------------------------------------------------

def _cv2_resize(img: np.ndarray, size, interpolation=None) -> np.ndarray:
    dw, dh = size                           # dsize = (width, height)
    sh, sw = img.shape[:2]
    zy, zx = dh / sh, dw / sw              # zy = row scale = height scale
    order = 3  # bicubic ≈ order 3 spline
    if img.ndim == 2:
        return _zoom(img, (zy, zx), order=order).astype(img.dtype)
    return _zoom(img, (zy, zx, 1), order=order).astype(img.dtype)


def _cv2_gaussian_blur(img: np.ndarray, ksize, sigma_x, sigma_y=0) -> np.ndarray:
    """Gaussian blur stub using scipy."""
    from scipy.ndimage import gaussian_filter as _gf
    sigma = sigma_x if sigma_x > 0 else (ksize[0] - 1) / 6.0
    return _gf(img.astype(np.float32), sigma=sigma).astype(img.dtype)


cv2_stub = types.ModuleType("cv2")
cv2_stub.INTER_CUBIC = 2
cv2_stub.INTER_LINEAR = 1
cv2_stub.COLOR_LAB2BGR = 56
cv2_stub.COLOR_BGR2RGB = 4
cv2_stub.resize = _cv2_resize
cv2_stub.GaussianBlur = _cv2_gaussian_blur
sys.modules["cv2"] = cv2_stub

# ---------------------------------------------------------------------------
# torch stub (only the bits used by experts.py at import time)
# ---------------------------------------------------------------------------

def _no_grad():
    """Decorator/context-manager stub."""
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
torch_stub.device = lambda x: x  # just return the string
torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    empty_cache=lambda: None,
)

nn_stub = types.ModuleType("torch.nn")
nn_stub.Module = object
torch_stub.nn = nn_stub

sys.modules["torch"] = torch_stub
sys.modules["torch.nn"] = nn_stub

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------

sys.path.insert(0, "/home/zaeem/Documents/domain-adaptive-colorization")

# Import only what the tests need (avoid pipeline.py which needs full torch)
from colorize.colorization_config import ColorizationConfig  # noqa: E402
from colorize.extraction import CropBox, RegionExtractor     # noqa: E402
from colorize.blending import Blender                        # noqa: E402
from colorize.regions import Region                          # noqa: E402

# ---------------------------------------------------------------------------
# Re-implement tests (copy of test_colorization.py logic, using unittest)
# ---------------------------------------------------------------------------

H, W = 100, 150


def make_region(x0, y0, x1, y1, label="sky", domain="nature") -> Region:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    box = np.array([x0, y0, x1, y1], dtype=float)
    return Region(label=label, score=0.9, box=box, mask=mask, domain=domain)


def default_cfg(**kwargs) -> ColorizationConfig:
    return ColorizationConfig(**kwargs)


class TestRegionExtractorCropBox(unittest.TestCase):

    def test_no_margin_returns_original_box(self):
        cfg = default_cfg(margin=0.0)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 30, 60, 70)
        crop = ex.get_crop_box(region)
        self.assertEqual(crop.x0, 20)
        self.assertEqual(crop.y0, 30)
        self.assertEqual(crop.x1, 60)
        self.assertEqual(crop.y1, 70)

    def test_margin_expands_box(self):
        cfg = default_cfg(margin=0.2)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(30, 20, 70, 60)  # w=40, h=40
        crop = ex.get_crop_box(region)
        self.assertEqual(crop.x0, 22)
        self.assertEqual(crop.y0, 12)
        self.assertEqual(crop.x1, 78)
        self.assertEqual(crop.y1, 68)

    def test_clip_at_left_top_boundary(self):
        cfg = default_cfg(margin=0.5)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(0, 0, 20, 20)
        crop = ex.get_crop_box(region)
        self.assertGreaterEqual(crop.x0, 0)
        self.assertGreaterEqual(crop.y0, 0)

    def test_clip_at_right_bottom_boundary(self):
        cfg = default_cfg(margin=0.5)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(120, 70, 150, 100)
        crop = ex.get_crop_box(region)
        self.assertLessEqual(crop.x1, W)
        self.assertLessEqual(crop.y1, H)

    def test_full_image_box_clipped(self):
        cfg = default_cfg(margin=0.3)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(0, 0, W, H)
        crop = ex.get_crop_box(region)
        self.assertEqual(crop.x0, 0)
        self.assertEqual(crop.y0, 0)
        self.assertEqual(crop.x1, W)
        self.assertEqual(crop.y1, H)

    def test_crop_dimensions_positive(self):
        cfg = default_cfg(margin=0.2)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(50, 40, 90, 80)
        crop = ex.get_crop_box(region)
        self.assertGreaterEqual(crop.width, 1)
        self.assertGreaterEqual(crop.height, 1)

    def test_near_boundary_clipped_correctly(self):
        cfg = default_cfg(margin=0.4)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(5, 5, 15, 15)
        crop = ex.get_crop_box(region)
        self.assertEqual(crop.x0, 1)
        self.assertEqual(crop.y0, 1)


class TestRegionExtractorInputOutput(unittest.TestCase):

    def test_extract_input_size(self):
        cfg = default_cfg(margin=0.1, input_size=64)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 20, 60, 60)
        L_full = np.random.rand(H, W).astype(np.float32) * 100
        L_resized, _ = ex.extract_input(L_full, region)
        self.assertEqual(L_resized.shape, (64, 64))

    def test_map_output_canvas_shape(self):
        cfg = default_cfg(margin=0.1, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 20, 60, 60)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        ab_pred = np.ones((32, 32, 2), dtype=np.float32)
        canvas = ex.map_output(ab_pred, crop)
        self.assertEqual(canvas.shape, (H, W, 2))

    def test_map_output_zero_outside_crop(self):
        cfg = default_cfg(margin=0.0, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(40, 30, 80, 70)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        ab_pred = np.ones((32, 32, 2), dtype=np.float32) * 5.0
        canvas = ex.map_output(ab_pred, crop)
        outside = np.ones((H, W), dtype=bool)
        outside[crop.y0:crop.y1, crop.x0:crop.x1] = False
        self.assertAlmostEqual(float(canvas[outside].sum()), 0.0, places=5)

    def test_map_output_nonzero_inside_crop(self):
        cfg = default_cfg(margin=0.0, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(40, 30, 80, 70)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        ab_pred = np.ones((32, 32, 2), dtype=np.float32) * 7.0
        canvas = ex.map_output(ab_pred, crop)
        inside = canvas[crop.y0:crop.y1, crop.x0:crop.x1]
        self.assertGreater(float(inside.sum()), 0.0)

    def test_binary_mask_box_mode(self):
        cfg = default_cfg(margin=0.0, region_shape="box")
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(10, 10, 50, 50)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        bm = ex.get_binary_mask(region, crop)
        self.assertTrue(bm[30, 30])
        self.assertFalse(bm[0, 0])

    def test_binary_mask_mask_mode(self):
        cfg = default_cfg(margin=0.0, region_shape="mask")
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(10, 10, 50, 50)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        bm = ex.get_binary_mask(region, crop)
        self.assertTrue(bm[30, 30])
        self.assertFalse(bm[5, 5])


class TestBlenderSigma(unittest.TestCase):

    def test_sigma_capped_by_sigma0(self):
        # ρ·min(w,h)/3 = 0.2·300/3 = 20 > σ_0=10, so sigma is capped at σ_0=10
        cfg = default_cfg(margin=0.2, smoothing_width=10.0)
        blender = Blender(cfg, (400, 500))  # large image so crop can be big
        big_crop = CropBox(0, 0, 500, 300)  # w=500, h=300 → 0.2·300/3=20 > 10
        sigma = blender.sigma_for_region(big_crop)
        self.assertAlmostEqual(sigma, 10.0, places=5)

    def test_sigma_formula_when_smaller(self):
        cfg = default_cfg(margin=0.2, smoothing_width=100.0)
        blender = Blender(cfg, (H, W))
        crop = CropBox(10, 10, 40, 30)  # w=30, h=20
        sigma = blender.sigma_for_region(crop)
        expected = 0.2 * 20 / 3.0
        self.assertAlmostEqual(sigma, expected, places=5)

    def test_sigma_zero_sigma0(self):
        cfg = default_cfg(margin=0.2, smoothing_width=0.0)
        blender = Blender(cfg, (H, W))
        crop = CropBox(10, 10, 60, 60)
        self.assertAlmostEqual(blender.sigma_for_region(crop), 0.0, places=5)


class TestBlenderSoftMask(unittest.TestCase):

    def _full_crop(self):
        return CropBox(0, 0, W, H)

    def test_soft_mask_shape(self):
        cfg = default_cfg()
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        self.assertEqual(M.shape, (H, W))

    def test_soft_mask_values_in_range(self):
        cfg = default_cfg(blending_on=True)
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        self.assertGreaterEqual(float(M.min()), 0.0)
        self.assertLessEqual(float(M.max()), 1.0 + 1e-6)

    def test_blending_off_gives_binary(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        np.testing.assert_array_equal(M, bm.astype(np.float32))

    def test_soft_mask_restricted_to_crop(self):
        cfg = default_cfg(blending_on=True)
        blender = Blender(cfg, (H, W))
        bm = np.ones((H, W), dtype=bool)
        crop = CropBox(20, 10, 80, 70)
        M = blender.soft_mask(bm, crop, sigma=5.0)
        self.assertAlmostEqual(float(M[:10, :].sum()), 0.0, places=5)
        self.assertAlmostEqual(float(M[70:, :].sum()), 0.0, places=5)
        self.assertAlmostEqual(float(M[:, :20].sum()), 0.0, places=5)
        self.assertAlmostEqual(float(M[:, 80:].sum()), 0.0, places=5)


class TestBlenderCombination(unittest.TestCase):

    def _canvas(self, v):
        return np.full((H, W, 2), v, dtype=np.float32)

    def _mask(self, y0, y1, x0, x1):
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def _crop(self, x0, y0, x1, y1):
        return CropBox(x0, y0, x1, y1)

    def test_single_region_gives_its_value(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab = self._canvas(42.0)
        bm = np.ones((H, W), dtype=bool)
        crop = self._crop(0, 0, W, H)
        result = blender.blend([ab], [bm], [crop])
        np.testing.assert_allclose(result, ab, atol=1e-5)

    def test_disjoint_regions_no_blending(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab_a = self._canvas(10.0)
        bm_a = self._mask(0, H, 0, W // 2)
        crop_a = self._crop(0, 0, W // 2, H)
        ab_b = self._canvas(20.0)
        bm_b = self._mask(0, H, W // 2, W)
        crop_b = self._crop(W // 2, 0, W, H)
        result = blender.blend([ab_a, ab_b], [bm_a, bm_b], [crop_a, crop_b])
        np.testing.assert_allclose(result[:, :W//2], 10.0, atol=1e-5)
        np.testing.assert_allclose(result[:, W//2:], 20.0, atol=1e-5)

    def test_weighted_average_at_overlap(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab_a = self._canvas(10.0)
        ab_b = self._canvas(30.0)
        bm = np.ones((H, W), dtype=bool)
        crop = self._crop(0, 0, W, H)
        result = blender.blend([ab_a, ab_b], [bm, bm], [crop, crop])
        np.testing.assert_allclose(result, 20.0, atol=1e-5)

    def test_division_by_zero_safety(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab = self._canvas(5.0)
        bm = self._mask(0, H // 2, 0, W // 2)
        crop = self._crop(0, 0, W // 2, H // 2)
        result = blender.blend([ab], [bm], [crop])
        uncovered = result[H // 2:, W // 2:]
        self.assertTrue(np.isfinite(uncovered).all())
        np.testing.assert_array_equal(uncovered, 0.0)

    def test_no_nan_or_inf_in_output(self):
        cfg = default_cfg(blending_on=True, smoothing_width=5.0)
        blender = Blender(cfg, (H, W))
        rng = np.random.default_rng(0)
        ab = rng.uniform(-50, 50, (H, W, 2)).astype(np.float32)
        bm = rng.random((H, W)) > 0.5
        crop = self._crop(10, 10, 120, 80)
        result = blender.blend([ab], [bm], [crop])
        self.assertTrue(np.isfinite(result).all())

    def test_three_disjoint_strips(self):
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        strips = [(0, 50), (50, 100), (100, 150)]
        values = [10.0, 20.0, 30.0]
        ab_list, bm_list, crop_list = [], [], []
        for (x0, x1), val in zip(strips, values):
            ab_list.append(self._canvas(val))
            bm_list.append(self._mask(0, H, x0, x1))
            crop_list.append(self._crop(x0, 0, x1, H))
        result = blender.blend(ab_list, bm_list, crop_list)
        for (x0, x1), val in zip(strips, values):
            np.testing.assert_allclose(result[:, x0:x1], val, atol=1e-5)


class TestColorizationConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = ColorizationConfig()
        self.assertAlmostEqual(cfg.margin, 0.2)
        self.assertAlmostEqual(cfg.smoothing_width, 10.0)
        self.assertTrue(cfg.blending_on)
        self.assertEqual(cfg.region_shape, "mask")
        self.assertEqual(cfg.input_size, 256)

    def test_invalid_region_shape(self):
        with self.assertRaises(ValueError):
            ColorizationConfig(region_shape="circle")

    def test_negative_margin(self):
        with self.assertRaises(ValueError):
            ColorizationConfig(margin=-0.1)

    def test_negative_smoothing(self):
        with self.assertRaises(ValueError):
            ColorizationConfig(smoothing_width=-5.0)

    def test_zero_input_size(self):
        with self.assertRaises(ValueError):
            ColorizationConfig(input_size=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

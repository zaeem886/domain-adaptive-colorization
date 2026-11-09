"""Unit tests for RegionExtractor and Blender.

Run with:  pytest tests/test_colorization.py -v

No GPU or model weights are required; torch is not imported anywhere here.
"""
from __future__ import annotations

import numpy as np
import pytest

from colorize.colorization_config import ColorizationConfig
from colorize.blending import Blender
from colorize.extraction import CropBox, RegionExtractor
from colorize.regions import Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 100, 150  # working image dimensions used throughout


def make_region(x0, y0, x1, y1, label="sky", domain="nature") -> Region:
    """Build a Region with a filled rectangular SAM mask."""
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    box = np.array([x0, y0, x1, y1], dtype=float)
    return Region(label=label, score=0.9, box=box, mask=mask, domain=domain)


def default_cfg(**kwargs) -> ColorizationConfig:
    return ColorizationConfig(**kwargs)


# ===========================================================================
# RegionExtractor tests
# ===========================================================================


class TestRegionExtractorCropBox:
    """Verify contextual cropping and boundary clipping."""

    def test_no_margin_returns_original_box(self):
        cfg = default_cfg(margin=0.0)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 30, 60, 70)
        crop = ex.get_crop_box(region)
        assert crop.x0 == 20
        assert crop.y0 == 30
        assert crop.x1 == 60
        assert crop.y1 == 70

    def test_margin_expands_box(self):
        """ρ=0.2 on a 40×40 box should add 8 pixels on each side."""
        cfg = default_cfg(margin=0.2)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(30, 20, 70, 60)  # w=40, h=40
        crop = ex.get_crop_box(region)
        # Expected: x0=30-8=22, x1=70+8=78, y0=20-8=12, y1=60+8=68
        assert crop.x0 == 22
        assert crop.y0 == 12
        assert crop.x1 == 78
        assert crop.y1 == 68

    def test_clip_at_left_top_boundary(self):
        """A box at the top-left corner must not produce negative coordinates."""
        cfg = default_cfg(margin=0.5)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(0, 0, 20, 20)
        crop = ex.get_crop_box(region)
        assert crop.x0 >= 0
        assert crop.y0 >= 0

    def test_clip_at_right_bottom_boundary(self):
        """A box at the bottom-right corner must stay within (W, H)."""
        cfg = default_cfg(margin=0.5)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(120, 70, 150, 100)
        crop = ex.get_crop_box(region)
        assert crop.x1 <= W
        assert crop.y1 <= H

    def test_full_image_box_is_clipped_to_image(self):
        """A bounding box that is already the full image should stay the full image."""
        cfg = default_cfg(margin=0.3)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(0, 0, W, H)
        crop = ex.get_crop_box(region)
        assert crop.x0 == 0
        assert crop.y0 == 0
        assert crop.x1 == W
        assert crop.y1 == H

    def test_crop_width_and_height_are_positive(self):
        """Crop must always have at least 1 pixel in each dimension."""
        cfg = default_cfg(margin=0.2)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(50, 40, 90, 80)
        crop = ex.get_crop_box(region)
        assert crop.width >= 1
        assert crop.height >= 1

    def test_near_boundary_region_clipped_correctly(self):
        """Region at x0=5 with margin=0.4 and w=10 → expanded x0 = 5-4=1 (≥0)."""
        cfg = default_cfg(margin=0.4)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(5, 5, 15, 15)  # w=10, h=10
        crop = ex.get_crop_box(region)
        # x0 - 0.4*10 = 5 - 4 = 1
        assert crop.x0 == 1
        assert crop.y0 == 1


class TestRegionExtractorInputOutput:
    """Verify extract_input dimensions and map_output coordinate mapping."""

    def test_extract_input_returns_correct_size(self):
        cfg = default_cfg(margin=0.1, input_size=64)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 20, 60, 60)
        L_full = np.random.rand(H, W).astype(np.float32) * 100
        L_resized, crop = ex.extract_input(L_full, region)
        assert L_resized.shape == (64, 64)

    def test_map_output_canvas_shape(self):
        cfg = default_cfg(margin=0.1, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(20, 20, 60, 60)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        ab_pred = np.ones((32, 32, 2), dtype=np.float32)
        canvas = ex.map_output(ab_pred, crop)
        assert canvas.shape == (H, W, 2)

    def test_map_output_zero_outside_crop(self):
        """Pixels outside the crop rectangle must remain zero."""
        cfg = default_cfg(margin=0.0, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(40, 30, 80, 70)   # w=40 h=40
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)

        ab_pred = np.ones((32, 32, 2), dtype=np.float32) * 5.0
        canvas = ex.map_output(ab_pred, crop)

        # Outside: sum should be zero
        outside = np.ones((H, W), dtype=bool)
        outside[crop.y0:crop.y1, crop.x0:crop.x1] = False
        assert canvas[outside].sum() == 0.0

    def test_map_output_nonzero_inside_crop(self):
        """Pixels inside the crop rectangle must carry the expert's output."""
        cfg = default_cfg(margin=0.0, input_size=32)
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(40, 30, 80, 70)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        ab_pred = np.ones((32, 32, 2), dtype=np.float32) * 7.0
        canvas = ex.map_output(ab_pred, crop)
        inside = canvas[crop.y0:crop.y1, crop.x0:crop.x1]
        assert inside.sum() > 0.0

    def test_binary_mask_box_mode(self):
        """In box mode the binary mask should equal the filled bounding box."""
        cfg = default_cfg(margin=0.0, region_shape="box")
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(10, 10, 50, 50)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        bm = ex.get_binary_mask(region, crop)
        # Filled box pixels: y∈[10,50), x∈[10,50)
        assert bm[30, 30] == True
        assert bm[0, 0] == False

    def test_binary_mask_mask_mode(self):
        """In mask mode the binary mask must match the SAM mask (clipped to crop)."""
        cfg = default_cfg(margin=0.0, region_shape="mask")
        ex = RegionExtractor(cfg, (H, W))
        region = make_region(10, 10, 50, 50)
        L_full = np.zeros((H, W), dtype=np.float32)
        _, crop = ex.extract_input(L_full, region)
        bm = ex.get_binary_mask(region, crop)
        # The SAM mask is True in [10:50, 10:50] – should match after restriction
        assert bm[30, 30] == True
        assert bm[5, 5] == False


# ===========================================================================
# Blender tests
# ===========================================================================


class TestBlenderSigma:
    """Dynamic σ_k calculation."""

    def test_sigma_capped_by_sigma0(self):
        """σ_k must not exceed σ_0.

        0.2 · min(500, 300) / 3 = 20 > σ_0=10, so min() clamps it to 10.
        """
        cfg = default_cfg(margin=0.2, smoothing_width=10.0)
        blender = Blender(cfg, (400, 500))
        big_crop = CropBox(0, 0, 500, 300)   # w=500, h=300 → formula=20 > σ_0
        sigma = blender.sigma_for_region(big_crop)
        assert sigma == pytest.approx(10.0)

    def test_sigma_uses_formula_when_smaller(self):
        """σ_k = ρ·min(w,h)/3 when this is smaller than σ_0."""
        cfg = default_cfg(margin=0.2, smoothing_width=100.0)
        blender = Blender(cfg, (H, W))
        # Crop: 30×20  → ρ·20/3 = 0.2·20/3 ≈ 1.333
        crop = CropBox(10, 10, 40, 30)   # w=30, h=20
        sigma = blender.sigma_for_region(crop)
        expected = 0.2 * 20 / 3.0
        assert sigma == pytest.approx(expected)

    def test_sigma_zero_sigma0(self):
        """σ_0 = 0 always gives σ_k = 0 (no smoothing)."""
        cfg = default_cfg(margin=0.2, smoothing_width=0.0)
        blender = Blender(cfg, (H, W))
        crop = CropBox(10, 10, 60, 60)
        sigma = blender.sigma_for_region(crop)
        assert sigma == pytest.approx(0.0)


class TestBlenderSoftMask:
    """Soft-mask construction and blending_on toggle."""

    def _full_crop(self) -> CropBox:
        return CropBox(0, 0, W, H)

    def test_soft_mask_shape(self):
        cfg = default_cfg()
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        assert M.shape == (H, W)

    def test_soft_mask_values_between_zero_and_one(self):
        cfg = default_cfg(blending_on=True)
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        assert M.min() >= 0.0
        assert M.max() <= 1.0 + 1e-6

    def test_blending_off_gives_binary_mask(self):
        """With blending_on=False, M_k must equal B_k (float version)."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        bm = np.zeros((H, W), dtype=bool)
        bm[20:60, 30:90] = True
        M = blender.soft_mask(bm, self._full_crop(), sigma=5.0)
        expected = bm.astype(np.float32)
        np.testing.assert_array_equal(M, expected)

    def test_soft_mask_restricted_to_crop(self):
        """Pixels outside the crop rectangle must be zero in M_k."""
        cfg = default_cfg(blending_on=True)
        blender = Blender(cfg, (H, W))
        bm = np.ones((H, W), dtype=bool)
        crop = CropBox(20, 10, 80, 70)
        M = blender.soft_mask(bm, crop, sigma=5.0)
        assert M[:10, :].sum() == pytest.approx(0.0)
        assert M[70:, :].sum() == pytest.approx(0.0)
        assert M[:, :20].sum() == pytest.approx(0.0)
        assert M[:, 80:].sum() == pytest.approx(0.0)


class TestBlenderCombination:
    """Weighted-average combination and division-by-zero safety."""

    def _make_canvas(self, value: float) -> np.ndarray:
        c = np.full((H, W, 2), value, dtype=np.float32)
        return c

    def _make_mask(self, y0, y1, x0, x1) -> np.ndarray:
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def _crop(self, x0, y0, x1, y1) -> CropBox:
        return CropBox(x0, y0, x1, y1)

    def test_single_region_gives_its_value(self):
        """With one region covering the whole canvas the output equals that canvas."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab = self._make_canvas(42.0)
        bm = np.ones((H, W), dtype=bool)
        crop = self._crop(0, 0, W, H)
        result = blender.blend([ab], [bm], [crop])
        np.testing.assert_allclose(result, ab, atol=1e-5)

    def test_disjoint_regions_no_blending(self):
        """Two disjoint masks with blending off → each pixel gets its region's value."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        # Region A: left half, value = 10
        ab_a = self._make_canvas(10.0)
        bm_a = self._make_mask(0, H, 0, W // 2)
        crop_a = self._crop(0, 0, W // 2, H)
        # Region B: right half, value = 20
        ab_b = self._make_canvas(20.0)
        bm_b = self._make_mask(0, H, W // 2, W)
        crop_b = self._crop(W // 2, 0, W, H)
        result = blender.blend([ab_a, ab_b], [bm_a, bm_b], [crop_a, crop_b])
        np.testing.assert_allclose(result[:, :W//2], 10.0, atol=1e-5)
        np.testing.assert_allclose(result[:, W//2:], 20.0, atol=1e-5)

    def test_weighted_average_at_overlap(self):
        """Equal-weight overlap of value-10 and value-30 must give value 20."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        ab_a = self._make_canvas(10.0)
        ab_b = self._make_canvas(30.0)
        # Both masks cover the full canvas → weights are equal → avg = 20
        bm_a = np.ones((H, W), dtype=bool)
        bm_b = np.ones((H, W), dtype=bool)
        crop_full = self._crop(0, 0, W, H)
        result = blender.blend([ab_a, ab_b], [bm_a, bm_b], [crop_full, crop_full])
        np.testing.assert_allclose(result, 20.0, atol=1e-5)

    def test_division_by_zero_safety(self):
        """Pixels covered by no mask must yield 0.0, not NaN or Inf."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        # Mask and canvas cover only the top-left quadrant
        ab = self._make_canvas(5.0)
        bm = self._make_mask(0, H // 2, 0, W // 2)
        crop = self._crop(0, 0, W // 2, H // 2)
        result = blender.blend([ab], [bm], [crop])
        # Bottom-right quadrant is uncovered → must be 0.0
        uncovered = result[H // 2:, W // 2:]
        assert np.isfinite(uncovered).all()
        np.testing.assert_array_equal(uncovered, 0.0)

    def test_no_nan_or_inf_in_output(self):
        """The blended output must never contain NaN or Inf values."""
        cfg = default_cfg(blending_on=True, smoothing_width=5.0)
        blender = Blender(cfg, (H, W))
        rng = np.random.default_rng(0)
        ab = rng.uniform(-50, 50, (H, W, 2)).astype(np.float32)
        bm = rng.random((H, W)) > 0.5
        crop = self._crop(10, 10, 120, 80)
        result = blender.blend([ab], [bm], [crop])
        assert np.isfinite(result).all()

    def test_three_regions_sum_to_correct_weighted_avg(self):
        """Three non-overlapping regions with known values and areas."""
        cfg = default_cfg(blending_on=False)
        blender = Blender(cfg, (H, W))
        # Divide the image into three vertical strips: 0:50, 50:100, 100:150
        strips = [(0, 50), (50, 100), (100, 150)]
        values = [10.0, 20.0, 30.0]
        ab_list, bm_list, crop_list = [], [], []
        for (x0, x1), val in zip(strips, values):
            ab_list.append(self._make_canvas(val))
            bm_list.append(self._make_mask(0, H, x0, x1))
            crop_list.append(self._crop(x0, 0, x1, H))
        result = blender.blend(ab_list, bm_list, crop_list)
        for (x0, x1), val in zip(strips, values):
            np.testing.assert_allclose(result[:, x0:x1], val, atol=1e-5)


# ===========================================================================
# ColorizationConfig tests
# ===========================================================================


class TestColorizationConfig:
    def test_defaults(self):
        cfg = ColorizationConfig()
        assert cfg.margin == pytest.approx(0.2)
        assert cfg.smoothing_width == pytest.approx(10.0)
        assert cfg.blending_on is True
        assert cfg.region_shape == "mask"
        assert cfg.input_size == 256

    def test_invalid_region_shape_raises(self):
        with pytest.raises(ValueError):
            ColorizationConfig(region_shape="circle")

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError):
            ColorizationConfig(margin=-0.1)

    def test_negative_smoothing_raises(self):
        with pytest.raises(ValueError):
            ColorizationConfig(smoothing_width=-5.0)

    def test_zero_input_size_raises(self):
        with pytest.raises(ValueError):
            ColorizationConfig(input_size=0)

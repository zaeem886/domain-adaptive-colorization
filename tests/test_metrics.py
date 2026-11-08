"""Unit tests for colorize.metrics (Person C).

No GPU, no model weights, and no COCO data are required.  All tests use
synthetic numpy arrays.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _rand_rgb(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    """Return a random uint8 RGB image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray_rgb(h: int = 32, w: int = 32, value: int = 128) -> np.ndarray:
    """Return a uniform gray uint8 RGB image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _colorful_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    """Return a strongly coloured image (pure red left half, pure blue right half)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, 0] = 255   # red
    img[:, w // 2 :, 2] = 255   # blue
    return img


# ---------------------------------------------------------------------------
# Minimal Region / Decomposition stubs (no model weights needed)
# ---------------------------------------------------------------------------

@dataclass
class _StubRegion:
    label: str
    score: float
    box: np.ndarray
    mask: np.ndarray
    domain: Optional[str] = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class _StubDecomposition:
    shape: tuple
    regions: List[_StubRegion]
    domains: tuple
    owner: np.ndarray
    domain_map: np.ndarray
    coverage: float
    domain_share: dict
    discarded: List[_StubRegion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from colorize.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lab_delta_e,
    compute_colorfulness,
    compute_miou,
    compute_pixel_coverage,
    compute_routing_accuracy,
    compute_confusion_matrix,
    compute_boundary_delta_e,
    compute_fid,
)
from colorize.results import (
    ResultsRecord,
    save_results,
    load_results,
    append_result,
    records_to_dataframe,
)


# ===========================================================================
# PSNR
# ===========================================================================

class TestComputePsnr:
    def test_identical_images_inf(self):
        img = _rand_rgb()
        result = compute_psnr(img, img.copy())
        assert math.isinf(result) and result > 0

    def test_different_images_finite_positive(self):
        img_true = _gray_rgb(value=128)
        img_pred = _gray_rgb(value=0)
        result = compute_psnr(img_true, img_pred)
        assert math.isfinite(result) and result > 0

    def test_known_mse(self):
        """MSE = 1 for a 1-pixel image → PSNR = 10·log10(255²/1) ≈ 48.13 dB."""
        img_true = np.array([[[100, 100, 100]]], dtype=np.uint8)
        img_pred = np.array([[[101, 100, 100]]], dtype=np.uint8)
        # MSE = (1² + 0 + 0) / 3 ≈ 0.333
        expected = 10.0 * math.log10(255.0 ** 2 / (1.0 / 3.0))
        assert abs(compute_psnr(img_true, img_pred) - expected) < 0.1

    def test_wrong_dtype_raises(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_psnr(img, img)

    def test_shape_mismatch_raises(self):
        img_a = _rand_rgb(16, 16)
        img_b = _rand_rgb(32, 32)
        with pytest.raises(ValueError):
            compute_psnr(img_a, img_b)


# ===========================================================================
# SSIM
# ===========================================================================

class TestComputeSsim:
    def test_identical_images_one(self):
        img = _rand_rgb(64, 64)
        result = compute_ssim(img, img.copy())
        assert abs(result - 1.0) < 1e-5, f"SSIM of identical images should be 1.0, got {result}"

    def test_different_images_less_than_one(self):
        img_true = _gray_rgb(value=0)
        img_pred = _gray_rgb(value=255)
        result = compute_ssim(img_true, img_pred)
        assert result < 1.0

    def test_range(self):
        img_true = _rand_rgb(seed=1)
        img_pred = _rand_rgb(seed=2)
        result = compute_ssim(img_true, img_pred)
        assert -1.0 <= result <= 1.0


# ===========================================================================
# Lab ΔE
# ===========================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("skimage"),
    reason="scikit-image not installed",
)
class TestComputeLabDeltaE:
    def test_identical_images_zero(self):
        img = _rand_rgb()
        result = compute_lab_delta_e(img, img.copy())
        assert result < 1e-6, f"ΔE of identical images should be ~0, got {result}"

    def test_different_images_positive(self):
        img_true = _gray_rgb(value=0)
        img_pred = _gray_rgb(value=255)
        result = compute_lab_delta_e(img_true, img_pred)
        assert result > 0

    def test_symmetry(self):
        img_a = _rand_rgb(seed=10)
        img_b = _rand_rgb(seed=11)
        # ΔE is symmetric
        assert abs(compute_lab_delta_e(img_a, img_b) - compute_lab_delta_e(img_b, img_a)) < 1e-5


# ===========================================================================
# Colourfulness
# ===========================================================================

class TestComputeColorfulness:
    def test_gray_image_low(self):
        img = _gray_rgb(value=128)
        result = compute_colorfulness(img)
        # R=G=B → rg=0, yb=0 → colourfulness=0
        assert result < 1.0, f"Gray image should have low colourfulness, got {result}"

    def test_colorful_higher_than_gray(self):
        gray = _gray_rgb()
        colorful = _colorful_rgb()
        assert compute_colorfulness(colorful) > compute_colorfulness(gray)

    def test_non_negative(self):
        img = _rand_rgb()
        assert compute_colorfulness(img) >= 0.0

    def test_pure_red(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[..., 0] = 255
        # rg = 255 - 0 = 255, yb = 0.5*255 - 0 = 127.5
        result = compute_colorfulness(img)
        assert result > 50.0


# ===========================================================================
# mIoU
# ===========================================================================

class TestComputeMiou:
    def test_perfect_prediction(self):
        gt = np.array([[0, 1], [2, 0]], dtype=np.int64)
        pred = gt.copy()
        result = compute_miou(pred, gt, num_classes=3)
        assert abs(result - 1.0) < 1e-6

    def test_all_wrong_two_classes(self):
        gt = np.array([[0, 0], [0, 0]], dtype=np.int64)
        pred = np.array([[1, 1], [1, 1]], dtype=np.int64)
        result = compute_miou(pred, gt, num_classes=2)
        # Class 0: intersection=0, union=4 → IoU=0
        # Class 1: intersection=0, union=4 → IoU=0
        assert result == 0.0

    def test_partial_overlap(self):
        gt = np.array([[0, 0, 1, 1]], dtype=np.int64)
        pred = np.array([[0, 1, 1, 1]], dtype=np.int64)
        result = compute_miou(pred, gt, num_classes=2)
        # class 0: pred={0}, gt={0,1} → intersection=1, union=2 → IoU=1/2
        # class 1: pred={1,2,3}, gt={2,3} → intersection=2, union=3 → IoU=2/3
        expected = (1 / 2 + 2 / 3) / 2  # ≈ 0.5833...
        assert abs(result - expected) < 1e-6

    def test_absent_classes_excluded(self):
        gt = np.zeros((4, 4), dtype=np.int64)
        pred = np.zeros((4, 4), dtype=np.int64)
        # Class 1 is absent from both → excluded
        result = compute_miou(pred, gt, num_classes=5)
        assert abs(result - 1.0) < 1e-6


# ===========================================================================
# Pixel coverage
# ===========================================================================

class TestComputePixelCoverage:
    def _make_decomp(self, shape, covered_fraction):
        H, W = shape
        total = H * W
        n_covered = int(total * covered_fraction)
        owner = np.full((H, W), -1, dtype=np.int32)
        owner.flat[:n_covered] = 0  # mark first N pixels as owned
        return _StubDecomposition(
            shape=shape,
            regions=[],
            domains=("fallback",),
            owner=owner,
            domain_map=np.zeros((H, W), dtype=np.int8),
            coverage=covered_fraction,
            domain_share={"fallback": 1.0},
        )

    def test_fully_covered(self):
        d = self._make_decomp((10, 10), 1.0)
        assert compute_pixel_coverage(d) == 1.0

    def test_half_covered(self):
        d = self._make_decomp((10, 10), 0.5)
        assert abs(compute_pixel_coverage(d) - 0.5) < 1e-6

    def test_no_coverage(self):
        d = self._make_decomp((8, 8), 0.0)
        assert compute_pixel_coverage(d) == 0.0


# ===========================================================================
# Routing accuracy
# ===========================================================================

class TestComputeRoutingAccuracy:
    def _make_regions(self, H, W):
        """Two non-overlapping regions: left half → people, right half → nature."""
        mask_left = np.zeros((H, W), dtype=bool)
        mask_left[:, : W // 2] = True
        mask_right = np.zeros((H, W), dtype=bool)
        mask_right[:, W // 2 :] = True
        r1 = _StubRegion(label="person", score=0.9, box=np.zeros(4), mask=mask_left, domain="people")
        r2 = _StubRegion(label="tree", score=0.8, box=np.zeros(4), mask=mask_right, domain="nature")
        return [r1, r2]

    def test_perfect_routing(self):
        H, W = 8, 8
        regions = self._make_regions(H, W)
        # GT: left half=0 (people), right half=1 (nature)
        gt = np.zeros((H, W), dtype=np.int64)
        gt[:, W // 2 :] = 1
        domain_order = ["people", "nature"]
        acc = compute_routing_accuracy(regions, gt, domain_order)
        assert abs(acc - 1.0) < 1e-6

    def test_fully_wrong(self):
        H, W = 8, 8
        regions = self._make_regions(H, W)
        # GT: left half=1 (nature), right half=0 (people) — opposite of predictions
        gt = np.ones((H, W), dtype=np.int64)
        gt[:, W // 2 :] = 0
        domain_order = ["people", "nature"]
        acc = compute_routing_accuracy(regions, gt, domain_order)
        assert acc == 0.0

    def test_empty_regions(self):
        gt = np.zeros((4, 4), dtype=np.int64)
        acc = compute_routing_accuracy([], gt, ["people"])
        assert acc == 0.0


# ===========================================================================
# Confusion matrix
# ===========================================================================

class TestComputeConfusionMatrix:
    def test_shape(self):
        H, W = 8, 8
        mask = np.zeros((H, W), dtype=bool)
        mask[:, :4] = True
        region = _StubRegion(label="person", score=0.9, box=np.zeros(4), mask=mask, domain="people")
        gt = np.zeros((H, W), dtype=np.int64)
        domain_order = ["people", "nature", "urban", "animals"]
        cm = compute_confusion_matrix([region], gt, domain_order)
        assert cm.shape == (4, 4)
        assert cm.dtype == np.int64

    def test_diagonal_for_correct_routing(self):
        H, W = 4, 4
        mask = np.ones((H, W), dtype=bool)
        region = _StubRegion(label="person", score=0.9, box=np.zeros(4), mask=mask, domain="people")
        gt = np.zeros((H, W), dtype=np.int64)  # all pixels → people (index 0)
        domain_order = ["people", "nature"]
        cm = compute_confusion_matrix([region], gt, domain_order)
        # All pixels: GT=people, pred=people → cm[0,0] = 16
        assert cm[0, 0] == H * W
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0

    def test_off_diagonal_for_wrong_routing(self):
        H, W = 4, 4
        mask = np.ones((H, W), dtype=bool)
        # Region predicted as 'nature' but GT is 'people'
        region = _StubRegion(label="tree", score=0.8, box=np.zeros(4), mask=mask, domain="nature")
        gt = np.zeros((H, W), dtype=np.int64)  # all pixels → people (index 0)
        domain_order = ["people", "nature"]
        cm = compute_confusion_matrix([region], gt, domain_order)
        # GT=people (0), pred=nature (1) → cm[0,1] = 16
        assert cm[0, 1] == H * W
        assert cm[0, 0] == 0


# ===========================================================================
# Boundary ΔE
# ===========================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("skimage"),
    reason="scikit-image not installed",
)
class TestComputeBoundaryDeltaE:
    def test_identical_images_zero(self):
        img = _rand_rgb(64, 64)
        result = compute_boundary_delta_e(img, img.copy(), sigma=1.0)
        assert result < 1e-5, f"Boundary ΔE of identical images should be ~0, got {result}"

    def test_different_images_positive(self):
        img_true = _rand_rgb(seed=5)
        img_pred = _rand_rgb(seed=6)
        result = compute_boundary_delta_e(img_true, img_pred, sigma=1.0)
        assert result >= 0.0


# ===========================================================================
# FID
# ===========================================================================

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("scipy"),
    reason="scipy not installed",
)
class TestComputeFid:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        feats = rng.standard_normal((100, 64)).astype(np.float64)
        fid = compute_fid(feats, feats.copy())
        # Identical distributions → FID should be very close to 0.
        # Allow a small tolerance for floating-point imprecision in sqrtm.
        assert abs(fid) < 1.0, f"FID of identical distributions should be ~0, got {fid}"

    def test_different_distributions_positive(self):
        rng = np.random.default_rng(0)
        real = rng.standard_normal((50, 32)).astype(np.float64)
        fake = rng.standard_normal((50, 32)).astype(np.float64) + 10.0  # large shift
        fid = compute_fid(real, fake)
        assert fid > 1.0, f"Expected large FID for very different distributions, got {fid}"

    def test_dimension_mismatch_raises(self):
        real = np.ones((10, 32), dtype=np.float64)
        fake = np.ones((10, 64), dtype=np.float64)
        with pytest.raises(ValueError):
            compute_fid(real, fake)


# ===========================================================================
# Results CSV (save → load roundtrip, append)
# ===========================================================================

class TestResultsCsv:
    def _make_records(self, n: int = 5) -> List[ResultsRecord]:
        return [
            ResultsRecord(
                experiment="E3",
                condition="ddcolor_t_whole",
                image_id=str(i),
                metric="psnr",
                value=float(20 + i),
            )
            for i in range(n)
        ]

    def test_save_load_roundtrip(self, tmp_path):
        records = self._make_records(5)
        csv_path = tmp_path / "results.csv"
        save_results(records, csv_path)
        loaded = load_results(csv_path)
        assert len(loaded) == len(records)
        for orig, back in zip(records, loaded):
            assert orig.experiment == back.experiment
            assert orig.condition == back.condition
            assert orig.image_id == back.image_id
            assert orig.metric == back.metric
            assert abs(orig.value - back.value) < 1e-9

    def test_save_empty(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        save_results([], csv_path)
        loaded = load_results(csv_path)
        assert loaded == []

    def test_append_creates_file(self, tmp_path):
        csv_path = tmp_path / "new.csv"
        rec = ResultsRecord("E3", "cond", "42", "ssim", 0.95)
        append_result(csv_path, rec)
        loaded = load_results(csv_path)
        assert len(loaded) == 1
        assert loaded[0].value == pytest.approx(0.95)

    def test_append_to_existing(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        records = self._make_records(3)
        save_results(records, csv_path)
        extra = ResultsRecord("E3", "new_cond", "99", "colorfulness", 42.0)
        append_result(csv_path, extra)
        loaded = load_results(csv_path)
        assert len(loaded) == 4
        assert loaded[-1].condition == "new_cond"
        assert loaded[-1].value == pytest.approx(42.0)

    def test_records_to_dataframe_returns_something(self):
        records = self._make_records(3)
        result = records_to_dataframe(records)
        # Either a DataFrame or a list of dicts
        assert len(result) == 3

    def test_float_precision_roundtrip(self, tmp_path):
        value = 3.14159265358979
        rec = ResultsRecord("E3", "c", "1", "psnr", value)
        csv_path = tmp_path / "precision.csv"
        save_results([rec], csv_path)
        loaded = load_results(csv_path)
        assert abs(loaded[0].value - value) < 1e-6


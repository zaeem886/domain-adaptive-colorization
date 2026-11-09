"""Unit tests for the E3 baseline without loading DDColor checkpoints."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import colorize.e3_baseline as e3
from colorize.results import load_results


def _write_rgb(path, height=12, width=16):
    y, x = np.indices((height, width))
    image = np.stack(
        [
            (x * 13 + y * 3) % 256,
            (y * 17 + 30) % 256,
            ((x + y) * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)
    return image


def test_resize_and_lab_helpers_preserve_expected_shapes():
    L = np.linspace(0.0, 100.0, 20, dtype=np.float32).reshape(4, 5)
    resized = e3._resize_L(L, 8, 10)
    assert resized.shape == (8, 10)
    assert resized.dtype == np.float32
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 100.0

    ab = np.zeros((4, 5, 2), dtype=np.float32)
    ab[..., 0] = 25.0
    restored = e3._resize_ab(ab, 8, 10)
    assert restored.shape == (8, 10, 2)
    rgb = e3._lab_to_rgb(resized, restored)
    assert rgb.shape == (8, 10, 3)
    assert rgb.dtype == np.uint8


def test_colorize_whole_image_rejects_invalid_inputs_before_model_loading():
    with pytest.raises(ValueError, match="2-D"):
        e3.colorize_whole_image(np.zeros((4, 4, 1)), "missing", "missing")
    with pytest.raises(ValueError, match="finite"):
        e3.colorize_whole_image(np.array([[np.nan]], dtype=np.float32), "missing", "missing")
    with pytest.raises(ValueError, match="positive"):
        e3.colorize_whole_image(np.zeros((4, 4)), "missing", "missing", input_size=0)


def test_load_image_l_returns_l_and_original_rgb(tmp_path):
    image_path = tmp_path / "input.png"
    original = _write_rgb(image_path)

    L, rgb = e3._load_image_L(image_path)

    assert L.shape == original.shape[:2]
    assert L.dtype == np.float32
    assert float(L.min()) >= 0.0
    assert float(L.max()) <= 100.0
    np.testing.assert_array_equal(rgb, original)


def test_run_e3_baseline_writes_colorfulness_and_gt_metrics(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    original = _write_rgb(image_path)

    def fake_colorize(L, checkpoint, ddcolor_path, device="cuda", input_size=256):
        assert L.ndim == 2
        assert checkpoint == "unused-checkpoint"
        assert ddcolor_path == "unused-ddcolor"
        predicted = original.copy()
        predicted[..., 0] = np.clip(predicted[..., 0].astype(np.int16) + 10, 0, 255)
        return predicted

    monkeypatch.setattr(e3, "colorize_whole_image", fake_colorize)
    output_csv = tmp_path / "results" / "e3.csv"

    records = e3.run_e3_baseline(
        image_paths=[("demo", image_path)],
        checkpoint="unused-checkpoint",
        ddcolor_path="unused-ddcolor",
        output_csv=output_csv,
        condition="ddcolor_t_whole",
        device="cpu",
    )

    assert output_csv.exists()
    assert {record.metric for record in records} == {
        "colorfulness", "psnr", "ssim", "lab_delta_e", "boundary_delta_e"
    }
    loaded = load_results(output_csv)
    assert len(loaded) == len(records)
    assert all(record.experiment == "E3" for record in loaded)
    assert all(record.condition == "ddcolor_t_whole" for record in loaded)


def test_run_e3_baseline_skips_unreadable_images(tmp_path):
    output_csv = tmp_path / "results.csv"
    records = e3.run_e3_baseline(
        image_paths=[(42, tmp_path / "does-not-exist.jpg")],
        checkpoint="unused-checkpoint",
        ddcolor_path="unused-ddcolor",
        output_csv=output_csv,
        condition="ddcolor_t_whole",
        device="cpu",
    )

    assert records == []
    assert not output_csv.exists()

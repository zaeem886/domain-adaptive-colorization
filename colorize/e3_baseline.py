"""Experiment E3: DDColor-T / DDColor-L whole-image baseline (Person C).

This module implements the E3 baseline: running a pretrained DDColor checkpoint
on the **whole image** (no Grounded-SAM decomposition, no routing) so that the
per-domain expert pipeline can be compared against a strong single-model baseline.

Main entry point: :func:`run_e3_baseline`.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore[import]
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image as PILImage  # type: ignore[import]
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Core colorisation helper (no decomposition)
# ---------------------------------------------------------------------------

def colorize_whole_image(
    L_full: np.ndarray,
    checkpoint: str,
    ddcolor_path: str,
    device: str = "cuda",
    input_size: int = 256,
) -> np.ndarray:
    """Colorise a full L-channel image with DDColor, without any decomposition.

    The image is resized to ``input_size × input_size``, passed through
    DDColor to obtain the ab channels, then the ab channels are resized back
    to the original resolution and merged with the original L channel to
    produce a uint8 RGB result.

    Parameters
    ----------
    L_full:
        2-D float32 array of shape ``(H, W)``, range [0, 100]
        (OpenCV CIE-Lab convention).
    checkpoint:
        Path to the DDColor ``.pth`` or ``.bin`` checkpoint file.
    ddcolor_path:
        Path to the root of the DDColor repository (must contain the
        ``ddcolor`` Python package).
    device:
        PyTorch device string (``"cuda"`` or ``"cpu"``).
    input_size:
        Spatial resolution fed to DDColor (256 for DDColor-T, 512 for
        DDColor-L).

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape ``(H, W, 3)``.

    Raises
    ------
    RuntimeError
        If PyTorch or the DDColor package cannot be imported.
    """
    L_full = np.asarray(L_full, dtype=np.float32)
    if L_full.ndim != 2:
        raise ValueError(f"L_full must be a 2-D array, got shape {L_full.shape}")
    if not np.isfinite(L_full).all():
        raise ValueError("L_full must contain only finite values")
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for E3 baseline colorisation. "
            "Install with: pip install torch"
        ) from exc

    # Add DDColor repo to path if needed
    ddcolor_path = str(ddcolor_path)
    if ddcolor_path not in sys.path:
        sys.path.insert(0, ddcolor_path)

    try:
        from ddcolor import DDColor  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'ddcolor' package is required. "
            "Clone https://github.com/piddnad/DDColor and pass its root via ddcolor_path."
        ) from exc

    H, W = L_full.shape

    # Determine backbone from input_size
    encoder = "convnext-t" if input_size <= 256 else "convnext-l"

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build model
    model = DDColor(
        encoder_name=encoder,
        decoder_name="MultiScaleColorDecoder",
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )
    ckpt = torch.load(str(checkpoint), map_location=dev)
    state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.warning(
            "Checkpoint %s did not match DDColor exactly "
            "(missing=%d, unexpected=%d keys).",
            checkpoint,
            len(missing),
            len(unexpected),
        )
    model.to(dev).eval()

    # Prepare input: L → grayscale-RGB at input_size
    L_resized = _resize_L(L_full, input_size, input_size)
    gray_rgb = _l_to_gray_rgb(L_resized)  # (S, S, 3) float32 [0,1]
    tensor = torch.from_numpy(gray_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)

    with torch.no_grad():
        ab_tensor = model(tensor)  # (1, 2, S, S)

    ab_small = ab_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (S, S, 2)

    # Resize ab back to original resolution
    ab_full = _resize_ab(ab_small, H, W)

    # Reconstruct Lab → RGB
    rgb = _lab_to_rgb(L_full, ab_full)
    return rgb


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _l_to_gray_rgb(L: np.ndarray) -> np.ndarray:
    """Convert L channel (H, W) in [0,100] to 3-channel grayscale-RGB in [0,1].

    Replicates the same logic used in ``experts._l_to_gray_rgb``.
    """
    lab = np.stack([L, np.zeros_like(L), np.zeros_like(L)], axis=-1).astype(np.float32)
    if _HAS_CV2:
        flag = getattr(cv2, "COLOR_LAB2RGB", getattr(cv2, "COLOR_Lab2RGB", None))
        rgb = cv2.cvtColor(lab, flag)
    else:
        try:
            from skimage.color import lab2rgb  # type: ignore[import]
            rgb = lab2rgb(lab).astype(np.float32)
        except ImportError:
            y = (L + 16.0) / 116.0
            delta = 6.0 / 29.0
            linear_y = np.where(y > delta, y ** 3, 3 * delta ** 2 * (y - 4.0 / 29.0))
            v = np.where(
                linear_y <= 0.0031308,
                12.92 * linear_y,
                1.055 * np.maximum(linear_y, 0.0) ** (1.0 / 2.4) - 0.055,
            )
            rgb = np.stack([v, v, v], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _resize_L(L: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize L channel to (h, w) using PIL (Lanczos)."""
    if not _HAS_PIL:
        # Fallback: nearest-neighbour via numpy index slicing
        from skimage.transform import resize as sk_resize  # type: ignore[import]
        return sk_resize(L, (h, w), anti_aliasing=True).astype(np.float32)
    img_pil = PILImage.fromarray(np.clip(L / 100.0 * 255.0, 0, 255).astype(np.uint8), mode="L")
    img_pil = img_pil.resize((w, h), PILImage.LANCZOS)
    return np.array(img_pil, dtype=np.float32) / 255.0 * 100.0


def _resize_ab(ab: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize ab channels (H, W, 2) to (h, w, 2)."""
    if _HAS_CV2:
        return cv2.resize(ab, (w, h), interpolation=cv2.INTER_LINEAR)
    if _HAS_PIL:
        a_pil = PILImage.fromarray(
            np.clip((ab[..., 0] + 128.0), 0, 255).astype(np.uint8), mode="L"
        ).resize((w, h), PILImage.BILINEAR)
        b_pil = PILImage.fromarray(
            np.clip((ab[..., 1] + 128.0), 0, 255).astype(np.uint8), mode="L"
        ).resize((w, h), PILImage.BILINEAR)
        a_full = np.array(a_pil, dtype=np.float32) - 128.0
        b_full = np.array(b_pil, dtype=np.float32) - 128.0
        return np.stack([a_full, b_full], axis=-1)
    # skimage fallback
    from skimage.transform import resize as sk_resize  # type: ignore[import]
    return sk_resize(ab, (h, w, 2), anti_aliasing=True).astype(np.float32)


def _lab_to_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Combine L (H, W) and ab (H, W, 2) → uint8 RGB (H, W, 3)."""
    lab = np.concatenate([L[..., np.newaxis], ab], axis=-1).astype(np.float32)
    if _HAS_CV2:
        flag = getattr(cv2, "COLOR_LAB2RGB", getattr(cv2, "COLOR_Lab2RGB", None))
        rgb = cv2.cvtColor(lab, flag)
    else:
        from skimage.color import lab2rgb  # type: ignore[import]
        rgb = lab2rgb(lab).astype(np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _load_image_L(image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image from *image_path* and return ``(L, rgb_uint8)`` pair.

    Returns
    -------
    L:
        Float32 array (H, W) in [0, 100] (OpenCV Lab convention).
    rgb_uint8:
        Original uint8 RGB, or None if the original image is grayscale.
    """
    from colorize.decomposition import load_gray  # type: ignore[import]
    from PIL import Image as PILImage  # type: ignore[import]

    # Try to load original RGB for GT metrics
    try:
        rgb_pil = PILImage.open(str(image_path)).convert("RGB")
        rgb_uint8 = np.array(rgb_pil, dtype=np.uint8)
    except Exception:
        rgb_uint8 = None

    gray_pil = load_gray(str(image_path))
    gray_arr = np.array(gray_pil, dtype=np.float32)  # (H, W) [0, 255]
    L = gray_arr / 255.0 * 100.0  # [0, 100]
    return L, rgb_uint8


# ---------------------------------------------------------------------------
# E3 baseline runner
# ---------------------------------------------------------------------------

def run_e3_baseline(
    image_paths: List[Tuple[Union[int, str], Union[str, Path]]],
    checkpoint: Union[str, Path],
    ddcolor_path: Union[str, Path],
    output_csv: Union[str, Path],
    condition: str,
    device: str = "cuda",
    input_size: int = 256,
    experiment: str = "E3",
    gt_root: Optional[Union[str, Path]] = None,
    fid_feature_records: Optional[dict] = None,
) -> List:
    """Run the E3 whole-image DDColor baseline and record metrics to CSV.

    For each image in *image_paths*, the function:

    1. Loads the grayscale L channel (and original RGB if ``gt_root`` is set).
    2. Runs DDColor on the **whole image** (no decomposition / routing).
    3. Computes colourfulness on the colourised output.
    4. If ``gt_root`` is provided, computes PSNR, SSIM, Lab ΔE, boundary ΔE,
       and LPIPS against the ground-truth image, and collects Inception
       features for the set-level FID calculation.
    5. Appends :class:`colorize.results.ResultsRecord` entries to *output_csv*.

    Parameters
    ----------
    image_paths:
        List of ``(image_id, path)`` tuples.  *image_id* may be an integer
        (COCO ID) or a string (e.g. ``"demo"``).
    checkpoint:
        Path to the DDColor ``.pth`` / ``.bin`` checkpoint.
    ddcolor_path:
        Path to the DDColor repository root (must contain the ``ddcolor``
        package).
    output_csv:
        Path to the results CSV file.  Rows are appended; the file is created
        with headers if it does not exist.
    condition:
        Condition label written to the CSV (e.g. ``"ddcolor_t_whole"``).
    device:
        PyTorch device string.
    input_size:
        DDColor input resolution (256 for DDColor-T, 512 for DDColor-L).
    experiment:
        Experiment label written to the CSV (default ``"E3"``).
    gt_root:
        Optional COCO ground-truth images root.  If provided and the GT file
        exists, PSNR / SSIM / ΔE metrics are computed.

    Returns
    -------
    list of ResultsRecord
        All records written during this call.
    """
    from colorize.results import ResultsRecord, append_result  # type: ignore[import]
    from colorize.metrics import (  # type: ignore[import]
        compute_colorfulness,
        compute_psnr,
        compute_ssim,
        compute_lab_delta_e,
        compute_boundary_delta_e,
        compute_lpips,
        compute_inception_features,
    )

    output_csv = Path(output_csv)
    checkpoint = str(checkpoint)
    ddcolor_path = str(ddcolor_path)
    all_records: List[ResultsRecord] = []

    for idx, (image_id, image_path) in enumerate(image_paths):
        image_path = Path(image_path)
        image_id_str = str(image_id)
        print(f"[E3] ({idx + 1}/{len(image_paths)}) Processing image {image_id_str} ...")

        # --- Load L channel ---
        try:
            L, rgb_orig = _load_image_L(image_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", image_path, exc)
            continue

        # --- Colorise whole image ---
        try:
            rgb_pred = colorize_whole_image(
                L, checkpoint, ddcolor_path, device=device, input_size=input_size
            )
        except Exception as exc:
            logger.warning("Colorisation failed for %s: %s", image_path, exc)
            continue

        # --- Colourfulness ---
        try:
            cf = compute_colorfulness(rgb_pred)
            rec = ResultsRecord(
                experiment=experiment,
                condition=condition,
                image_id=image_id_str,
                metric="colorfulness",
                value=cf,
            )
            append_result(output_csv, rec)
            all_records.append(rec)
        except Exception as exc:
            logger.warning("Colourfulness computation failed: %s", exc)

        # --- GT metrics (optional) ---
        gt_rgb = _get_gt_rgb(image_id, gt_root, rgb_orig)
        if gt_rgb is not None:
            # Resize prediction to GT size if needed
            gt_h, gt_w = gt_rgb.shape[:2]
            pred_h, pred_w = rgb_pred.shape[:2]
            if (pred_h, pred_w) != (gt_h, gt_w):
                rgb_pred_resized = _resize_rgb(rgb_pred, gt_h, gt_w)
            else:
                rgb_pred_resized = rgb_pred

            metric_fns = {
                "psnr": compute_psnr,
                "ssim": compute_ssim,
                "lab_delta_e": compute_lab_delta_e,
                "boundary_delta_e": compute_boundary_delta_e,
                "lpips": lambda a, b: compute_lpips(a, b, device=device),
            }
            for metric_name, fn in metric_fns.items():
                try:
                    value = fn(gt_rgb, rgb_pred_resized)
                    if not np.isfinite(value):
                        raise RuntimeError(
                            f"Metric '{metric_name}' returned a non-finite value; "
                            "check its optional dependencies"
                        )
                    rec = ResultsRecord(
                        experiment=experiment,
                        condition=condition,
                        image_id=image_id_str,
                        metric=metric_name,
                        value=float(value),
                    )
                    append_result(output_csv, rec)
                    all_records.append(rec)
                except Exception as exc:
                    if metric_name == "lpips":
                        raise RuntimeError(
                            "E3 LPIPS evaluation failed; verify that torch and lpips are installed"
                        ) from exc
                    logger.warning("Metric '%s' failed for %s: %s", metric_name, image_path, exc)

            if fid_feature_records is not None:
                real_features = compute_inception_features(gt_rgb, device=device)
                fake_features = compute_inception_features(rgb_pred_resized, device=device)
                fid_feature_records[image_id_str] = (real_features, fake_features)

    print(f"[E3] Done. {len(all_records)} records written to {output_csv}")
    return all_records


# ---------------------------------------------------------------------------
# Private GT helpers
# ---------------------------------------------------------------------------

def _get_gt_rgb(
    image_id: Union[int, str],
    gt_root: Optional[Union[str, Path]],
    rgb_orig: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return GT uint8 RGB for *image_id*, or ``None`` if unavailable."""
    if gt_root is not None and str(image_id).isdigit():
        from colorize.test_sets import get_coco_image_path  # type: ignore[import]
        from PIL import Image as PILImage  # type: ignore[import]
        try:
            gt_path = get_coco_image_path(int(image_id), gt_root)
            gt_rgb = np.array(PILImage.open(gt_path).convert("RGB"), dtype=np.uint8)
            return gt_rgb
        except Exception:
            pass

    # Fall back to the original RGB loaded alongside the input
    return rgb_orig


def _resize_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a uint8 RGB image to (h, w, 3)."""
    if _HAS_CV2:
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if _HAS_PIL:
        pil = PILImage.fromarray(rgb).resize((w, h), PILImage.BILINEAR)
        return np.array(pil, dtype=np.uint8)
    from skimage.transform import resize as sk_resize  # type: ignore[import]
    return (sk_resize(rgb.astype(np.float32) / 255.0, (h, w, 3)) * 255.0).astype(np.uint8)


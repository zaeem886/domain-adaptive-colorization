"""Blending and Reconstruction.

Implements the soft-mask weighted combination formula from the report::

    M_k = (G_{σ_k} * B_k) · 1_{C_k}
    σ_k = min(σ_0, ρ · min(w_k, h_k) / 3)
    â_b(x) = Σ_k M_k(x) â_b_k(x) / Σ_k M_k(x)

The final Lab image is converted to RGB and clipped to [0, 255].

No torch dependency: everything is NumPy / SciPy / OpenCV so the blender
can be unit-tested cheaply.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as _zoom

from .colorization_config import ColorizationConfig
from .extraction import CropBox, RegionExtractor
from .regions import Region


class Blender:
    """Combine per-region ab predictions into one full-image chrominance map.

    Parameters
    ----------
    cfg:
        Pipeline configuration.
    image_shape:
        ``(H, W)`` of the working image.
    """

    def __init__(self, cfg: ColorizationConfig, image_shape: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.H, self.W = image_shape

    # ------------------------------------------------------------------
    # σ_k formula
    # ------------------------------------------------------------------

    def sigma_for_region(self, crop: CropBox) -> float:
        """Compute the per-region smoothing width.

        .. math::

            \\sigma_k = \\min\\!\\left(\\sigma_0,\\; \\frac{\\rho \\cdot \\min(w_k, h_k)}{3}\\right)

        Parameters
        ----------
        crop:
            The contextual crop for region *k* (after margin expansion).
        """
        rho = self.cfg.margin
        sigma_0 = self.cfg.smoothing_width
        min_dim = min(crop.width, crop.height)
        return float(min(sigma_0, rho * min_dim / 3.0))

    # ------------------------------------------------------------------
    # Soft-mask construction
    # ------------------------------------------------------------------

    def soft_mask(self, binary_mask: np.ndarray, crop: CropBox, sigma: float) -> np.ndarray:
        """Build M_k from B_k.

        .. math::

            M_k = (G_{\\sigma_k} * B_k) \\cdot \\mathbf{1}_{C_k}

        When ``blending_on`` is False the Gaussian step is skipped and the
        raw binary mask is returned (still restricted to the crop rectangle).

        The Gaussian is applied only on the crop patch ``C_k`` using an odd
        kernel size ``ksize = int(round(sigma_k * 6)) | 1`` to ensure symmetry
        and strict spatial support restriction as per the formula.

        Parameters
        ----------
        binary_mask:
            Boolean array ``(H, W)`` — the region mask B_k.
        crop:
            Crop rectangle used to restrict M_k to 1_{C_k}.
        sigma:
            Standard deviation for the Gaussian filter (σ_k).

        Returns
        -------
        M_k:
            Float32 array ``(H, W)`` with values in [0, 1].
        """
        # Always start with a zero canvas; only the crop region is populated.
        mask_out = np.zeros((self.H, self.W), dtype=np.float32)

        # Extract the binary mask patch inside C_k.
        patch = binary_mask[crop.y0:crop.y1, crop.x0:crop.x1].astype(np.float32)

        if self.cfg.blending_on and sigma > 0:
            # Odd kernel size: ksize = int(round(sigma_k * 6)) | 1
            # The bitwise-OR with 1 guarantees the kernel is odd.
            ksize = int(round(sigma * 6)) | 1
            ksize = max(ksize, 1)  # guard against zero for tiny sigma

            if cv2 is not None and hasattr(cv2, 'GaussianBlur'):
                # cv2.GaussianBlur operates only on the crop patch → strict C_k support.
                blurred_patch = cv2.GaussianBlur(patch, (ksize, ksize), sigma)
            else:
                # scipy fallback: apply within the crop patch, then clip.
                blurred_patch = gaussian_filter(patch, sigma=sigma)
        else:
            blurred_patch = patch

        # Place blurred patch back at crop coordinates → enforces 1_{C_k}.
        mask_out[crop.y0:crop.y1, crop.x0:crop.x1] = blurred_patch
        return mask_out

    # ------------------------------------------------------------------
    # Main blend method
    # ------------------------------------------------------------------

    def blend(
        self,
        ab_canvases: Sequence[np.ndarray],
        binary_masks: Sequence[np.ndarray],
        crops: Sequence[CropBox],
    ) -> np.ndarray:
        """Compute the normalised weighted sum of all ab canvases.

        The list must include the fallback region (index 0) first, followed by
        the domain regions in any order.

        Parameters
        ----------
        ab_canvases:
            One ``(H, W, 2)`` float32 array per region (including fallback).
        binary_masks:
            One ``(H, W)`` bool array per region.
        crops:
            One :class:`CropBox` per region.

        Returns
        -------
        ab_blended:
            ``(H, W, 2)`` float32.
        """
        assert len(ab_canvases) == len(binary_masks) == len(crops), (
            "ab_canvases, binary_masks, and crops must have the same length."
        )

        numerator = np.zeros((self.H, self.W, 2), dtype=np.float64)
        denominator = np.zeros((self.H, self.W), dtype=np.float64)

        for ab, bm, crop in zip(ab_canvases, binary_masks, crops):
            sigma = self.sigma_for_region(crop)
            M = self.soft_mask(bm, crop, sigma).astype(np.float64)
            numerator += M[:, :, None] * ab.astype(np.float64)
            denominator += M

        # Division-by-zero safety: where no mask covers a pixel, output 0.
        safe_denom = np.where(denominator > 0, denominator, 1.0)
        ab_blended = (numerator / safe_denom[:, :, None]).astype(np.float32)
        # Force any genuinely uncovered pixel to 0.
        ab_blended[denominator == 0] = 0.0
        return ab_blended


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class Reconstructor:
    """Recombine L + ab → RGB.

    Parameters
    ----------
    original_shape:
        ``(H_orig, W_orig)`` of the image *before* any pipeline downscaling.
        If the working image was not downscaled this equals the working shape.
    working_shape:
        ``(H, W)`` of the working (possibly downscaled) image.
    """

    def __init__(
        self,
        original_shape: Tuple[int, int],
        working_shape: Tuple[int, int],
    ) -> None:
        self.original_shape = original_shape  # (H_orig, W_orig)
        self.working_shape = working_shape    # (H, W)

    def reconstruct(
        self,
        L0: np.ndarray,
        ab_blended: np.ndarray,
    ) -> np.ndarray:
        """Convert blended Lab to clipped uint8 RGB.

        Parameters
        ----------
        L0:
            Original L channel at **original** resolution, shape
            ``(H_orig, W_orig)``, range [0, 100].
        ab_blended:
            Blended ab map at **working** resolution, shape ``(H, W, 2)``,
            range [-128, 127].

        Returns
        -------
        rgb:
            ``(H_orig, W_orig, 3)`` uint8 RGB image.
        """
        H_orig, W_orig = self.original_shape
        H_work, W_work = self.working_shape

        # Upsample ab back to original resolution if needed.
        # cv2.resize dsize = (width, height); NumPy shape is (H, W) = (height, width).
        if (H_work, W_work) != (H_orig, W_orig):
            if cv2 is not None:
                ab_full = cv2.resize(
                    ab_blended,
                    (W_orig, H_orig),   # cv2: (width, height)
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                zy = H_orig / H_work   # row scale = height scale
                zx = W_orig / W_work   # col scale = width  scale
                ab_full = _zoom(ab_blended, (zy, zx, 1), order=1).astype(ab_blended.dtype)
        else:
            ab_full = ab_blended

        # Stack L (H, W, 1) + ab (H, W, 2) → Lab (H, W, 3).
        # Convention: L in [0, 100], a/b in [-128, 127].
        Lab = np.concatenate(
            [L0[:, :, None].astype(np.float32), ab_full], axis=2
        )

        # _lab_to_rgb handles the [0,255] uint8 scaling internally for cv2,
        # and passes float Lab directly to skimage when available.
        rgb = _lab_to_rgb(Lab)
        return rgb


def _lab_to_rgb(Lab: np.ndarray) -> np.ndarray:
    """Convert a float32 Lab image (OpenCV range) to clipped uint8 RGB.

    Uses ``skimage.color.lab2rgb`` when available, otherwise falls back to
    OpenCV's ``cvtColor`` with the standard uint8 encoding trick.

    Parameters
    ----------
    Lab:
        Shape ``(H, W, 3)``, L in [0, 100], a/b in [-128, 127].

    Returns
    -------
    rgb:
        Shape ``(H, W, 3)``, uint8, channels R G B.
    """
    try:
        from skimage.color import lab2rgb  # type: ignore[import]

        rgb_float = lab2rgb(Lab)              # → [0, 1] float64
        rgb = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
        return rgb

    except ImportError:
        pass

    # Fallback: OpenCV uint8 Lab encoding.
    # OpenCV's 8-bit Lab convention:
    #   L_8bit = L * 255/100  → maps [0,100] to [0,255]
    #   a_8bit = a + 128      → maps [-128,127] to [0,255]; neutral grey = 128
    #   b_8bit = b + 128      → same
    Lab_u8 = np.empty((Lab.shape[0], Lab.shape[1], 3), dtype=np.uint8)
    Lab_u8[:, :, 0] = np.clip(Lab[:, :, 0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
    Lab_u8[:, :, 1] = np.clip(Lab[:, :, 1] + 128.0, 0, 255).astype(np.uint8)
    Lab_u8[:, :, 2] = np.clip(Lab[:, :, 2] + 128.0, 0, 255).astype(np.uint8)

    if cv2 is not None:
        color_flag = getattr(cv2, "COLOR_LAB2BGR", getattr(cv2, "COLOR_Lab2BGR", None))
        if color_flag is not None:
            bgr = cv2.cvtColor(Lab_u8, color_flag)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb

    # Pure numpy/math Lab->sRGB fallback if neither skimage nor cv2 is available
    L = Lab[:, :, 0]
    a = Lab[:, :, 1]
    b = Lab[:, :, 2]

    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0

    def f_inv(t):
        delta = 6.0 / 29.0
        return np.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - 4.0 / 29.0))

    # D65 reference white
    X = 0.95047 * f_inv(x)
    Y = 1.00000 * f_inv(y)
    Z = 1.08883 * f_inv(z)

    # XYZ to linear sRGB
    R_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 0) ** (1.0 / 2.4)) - 0.055)

    R = np.clip(gamma(R_lin) * 255.0, 0, 255).astype(np.uint8)
    G = np.clip(gamma(G_lin) * 255.0, 0, 255).astype(np.uint8)
    B = np.clip(gamma(B_lin) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)

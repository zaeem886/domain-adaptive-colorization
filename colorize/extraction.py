"""Region Extraction: contextual cropping, resizing, and output mapping.

Report section 6.x — the bridge between Region objects (produced by Person A)
and the per-region inputs/outputs of each expert model.

No torch dependency here: all operations are pure NumPy / OpenCV so this module
can be unit-tested without a GPU.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from scipy.ndimage import zoom as _zoom

from .colorization_config import ColorizationConfig
from .regions import Region


class CropBox(NamedTuple):
    """Pixel-aligned, image-clipped crop rectangle."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _bicubic_resize(arr: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
    """Resize *arr* to target dimensions.

    Parameters
    ----------
    arr:
        Input array, shape ``(H, W)`` or ``(H, W, C)``.
    dsize:
        ``(target_width, target_height)`` — matches ``cv2.resize`` convention.
        The underlying NumPy / scipy fallback transposes this correctly.
    """
    target_w, target_h = dsize          # unpack explicitly to prevent confusion
    if cv2 is not None:
        # cv2.resize expects (width, height) — matches our dsize convention.
        return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    # scipy fallback: zoom factors are (row_scale, col_scale) = (h_scale, w_scale)
    src_h, src_w = arr.shape[:2]
    zy = target_h / src_h               # row axis  = height
    zx = target_w / src_w               # col axis  = width
    if arr.ndim == 2:
        return _zoom(arr, (zy, zx), order=3).astype(arr.dtype)
    return _zoom(arr, (zy, zx, 1), order=3).astype(arr.dtype)


class RegionExtractor:
    """Handles all coordinate arithmetic for one region crop.

    Parameters
    ----------
    cfg:
        A :class:`ColorizationConfig` instance.
    image_shape:
        ``(H, W)`` of the full working image (after any downscaling by
        :func:`~colorize.decomposition.load_gray`).
    """

    def __init__(self, cfg: ColorizationConfig, image_shape: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.H, self.W = image_shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_crop_box(self, region: Region) -> CropBox:
        """Compute the contextual crop rectangle for *region*.

        For a bounding box (x0, y0, x1, y1) with width w and height h the
        expanded rectangle is::

            x_lo = x0 - ρ·w,  x_hi = x1 + ρ·w
            y_lo = y0 - ρ·h,  y_hi = y1 + ρ·h

        Coordinates are clipped to [0, W) × [0, H) and rounded to integers.
        """
        x0, y0, x1, y1 = region.box.astype(float)
        w = x1 - x0
        h = y1 - y0
        rho = self.cfg.margin

        cx0 = int(np.round(np.clip(x0 - rho * w, 0, self.W - 1)))
        cy0 = int(np.round(np.clip(y0 - rho * h, 0, self.H - 1)))
        cx1 = int(np.round(np.clip(x1 + rho * w, cx0 + 1, self.W)))
        cy1 = int(np.round(np.clip(y1 + rho * h, cy0 + 1, self.H)))

        return CropBox(cx0, cy0, cx1, cy1)

    def extract_input(self, L_full: np.ndarray, region: Region) -> Tuple[np.ndarray, CropBox]:
        """Extract and resize the luminance crop for *region*.

        Parameters
        ----------
        L_full:
            Full-resolution L channel, shape ``(H, W)``, dtype float32,
            range [0, 100] (OpenCV Lab convention).

        Returns
        -------
        L_resized:
            Shape ``(S, S)``, bicubic-interpolated crop.
        crop:
            The :class:`CropBox` used (needed later by :meth:`map_output`).
        """
        S = self.cfg.input_size
        crop = self.get_crop_box(region)
        # NumPy slicing: [row_start:row_end, col_start:col_end] = [y0:y1, x0:x1]
        L_crop = L_full[crop.y0:crop.y1, crop.x0:crop.x1]
        # cv2.resize dsize = (width, height) — both S, so order doesn't matter here.
        L_resized = _bicubic_resize(L_crop, (S, S))
        return L_resized, crop

    def map_output(
        self,
        ab_pred: np.ndarray,
        crop: CropBox,
    ) -> np.ndarray:
        """Map an S×S ab prediction back to full-image coordinates.

        Steps
        -----
        1. Bicubic-resize *ab_pred* from ``(S, S, 2)`` back to the crop
           dimensions ``(crop.height, crop.width, 2)``.
        2. Place the result into a zero-initialised canvas of shape
           ``(H, W, 2)``.

        Parameters
        ----------
        ab_pred:
            Expert output, shape ``(S, S, 2)``, channels [a, b].

        Returns
        -------
        canvas:
            Shape ``(H, W, 2)``, zero everywhere except inside *crop*.
        """
        # crop dimensions in NumPy convention: height = y1-y0 rows, width = x1-x0 cols.
        crop_h = crop.height   # number of rows  → target_height for resize
        crop_w = crop.width    # number of cols  → target_width  for resize
        # _bicubic_resize dsize = (target_width, target_height)
        ab_crop = _bicubic_resize(ab_pred, (crop_w, crop_h))
        canvas = np.zeros((self.H, self.W, 2), dtype=np.float32)
        # Place using NumPy [row, col] = [y0:y1, x0:x1] slicing.
        canvas[crop.y0:crop.y1, crop.x0:crop.x1] = ab_crop
        return canvas

    def get_binary_mask(self, region: Region, crop: CropBox) -> np.ndarray:
        """Return the binary mask B_k at full image resolution.

        When ``cfg.region_shape == "mask"`` the SAM mask is used directly.
        When ``cfg.region_shape == "box"``  a filled rectangle is used instead.

        The mask is then restricted to the crop rectangle (the two should
        already agree for sane inputs, but this avoids accidental leakage).
        """
        H, W = self.H, self.W
        if self.cfg.region_shape == "box":
            mask = np.zeros((H, W), dtype=bool)
            x0, y0, x1, y1 = (int(v) for v in region.box)
            mask[y0:y1, x0:x1] = True
        else:
            mask = region.mask.astype(bool)

        # Restrict to the crop rectangle so B_k · 1_{C_k} = B_k
        restricted = np.zeros((H, W), dtype=bool)
        restricted[crop.y0:crop.y1, crop.x0:crop.x1] = (
            mask[crop.y0:crop.y1, crop.x0:crop.x1]
        )
        return restricted

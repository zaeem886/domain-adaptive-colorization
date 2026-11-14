"""Evaluation metrics for domain-adaptive colorization (Person C).

All functions are pure NumPy/SciPy — no GPU required.
FID and LPIPS use optional libraries (torch, lpips) and degrade gracefully
to NaN when those libraries are unavailable.

Metric summary
--------------
* compute_psnr            — Peak Signal-to-Noise Ratio (dB)
* compute_ssim            — Structural Similarity Index Measure
* compute_lab_delta_e     — Mean per-pixel CIEDE2000 ΔE in Lab space
* compute_boundary_delta_e— ΔE restricted to image edges
* compute_colorfulness    — Hasler & Süsstrunk (2003) colorfulness
* compute_lpips           — Learned Perceptual Image Patch Similarity
* compute_fid             — Fréchet Inception Distance from feature arrays
* compute_miou            — Mean Intersection-over-Union over classes
* compute_pixel_coverage  — Fraction of pixels owned by named regions
* compute_routing_accuracy— Fraction of pixels where routing domain == GT domain
* compute_confusion_matrix— Confusion matrix (predicted vs. GT domain)
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from skimage import color as skcolor  # type: ignore[import]
    from skimage import metrics as skmetrics  # type: ignore[import]
    from skimage import filters as skfilters  # type: ignore[import]
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from scipy import linalg as splinalg  # type: ignore[import]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_LPIPS_MODELS = {}
_INCEPTION_MODELS = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_uint8_rgb(arr: np.ndarray, name: str = "image") -> None:
    """Raise ValueError if *arr* is not a uint8 (H, W, 3) array."""
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {arr.shape}")


def _to_float_lab(img_rgb: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB image to float Lab via skimage."""
    if not _HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for Lab-based metrics. "
            "Install it with: pip install scikit-image"
        )
    rgb_float = img_rgb.astype(np.float32) / 255.0
    return skcolor.rgb2lab(rgb_float)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two uint8 RGB images.

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8.
    img_pred:
        Predicted/colorized image, same shape and dtype.

    Returns
    -------
    float
        PSNR in decibels.  Returns ``float('inf')`` when the images are
        identical (MSE == 0).
    """
    _check_uint8_rgb(img_true, "img_true")
    _check_uint8_rgb(img_pred, "img_pred")

    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Shape mismatch: img_true={img_true.shape}, img_pred={img_pred.shape}"
        )

    mse = np.mean((img_true.astype(np.float64) - img_pred.astype(np.float64)) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10(255.0 ** 2 / mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Compute mean Structural Similarity Index Measure (SSIM).

    Uses ``skimage.metrics.structural_similarity`` when available, otherwise
    falls back to a minimal manual implementation (multichannel mean).

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8.
    img_pred:
        Predicted/colorized image, same shape and dtype.

    Returns
    -------
    float
        SSIM in ``[−1, 1]``; 1.0 for identical images.
    """
    _check_uint8_rgb(img_true, "img_true")
    _check_uint8_rgb(img_pred, "img_pred")

    if _HAS_SKIMAGE:
        # channel_axis added in skimage 0.19; fall back to multichannel kwarg
        try:
            return float(
                skmetrics.structural_similarity(
                    img_true, img_pred, channel_axis=2, data_range=255
                )
            )
        except TypeError:
            return float(
                skmetrics.structural_similarity(
                    img_true, img_pred, multichannel=True, data_range=255
                )
            )

    # --- manual fallback (no skimage) ---
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = img_true.astype(np.float64)
    b = img_pred.astype(np.float64)
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a = np.var(a)
    sigma_b = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / den)


# ---------------------------------------------------------------------------
# Lab ΔE (CIEDE2000)
# ---------------------------------------------------------------------------

def compute_lab_delta_e(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Compute mean per-pixel CIEDE2000 ΔE in Lab colour space.

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8 RGB.
    img_pred:
        Predicted/colorized image, same shape and dtype.

    Returns
    -------
    float
        Mean ΔE₀₀ across all pixels.  0.0 for identical images.

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    """
    _check_uint8_rgb(img_true, "img_true")
    _check_uint8_rgb(img_pred, "img_pred")

    lab_true = _to_float_lab(img_true)
    lab_pred = _to_float_lab(img_pred)

    delta_e = skcolor.deltaE_ciede2000(lab_true, lab_pred)
    return float(np.mean(delta_e))


# ---------------------------------------------------------------------------
# Boundary ΔE
# ---------------------------------------------------------------------------

def compute_boundary_delta_e(
    img_true: np.ndarray,
    img_pred: np.ndarray,
    sigma: float = 2.0,
) -> float:
    """Compute ΔE restricted to edge pixels detected by a Sobel filter.

    Edges are detected on the grayscale version of *img_true* using the Sobel
    operator after Gaussian smoothing.  The returned value is the mean CIEDE2000
    ΔE over the pixels whose Sobel-gradient magnitude exceeds the mean gradient
    (top ~50 % of edge strength).

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8 RGB.
    img_pred:
        Predicted/colorized image, same shape and dtype.
    sigma:
        Standard deviation for Gaussian pre-smoothing before edge detection.

    Returns
    -------
    float
        Mean boundary ΔE₀₀.  0.0 for identical images.

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    """
    _check_uint8_rgb(img_true, "img_true")
    _check_uint8_rgb(img_pred, "img_pred")

    if not _HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for boundary ΔE. "
            "Install it with: pip install scikit-image"
        )

    # Grayscale for edge detection
    gray = skcolor.rgb2gray(img_true.astype(np.float32) / 255.0)
    if sigma > 0:
        from skimage.filters import gaussian  # type: ignore[import]
        gray = gaussian(gray, sigma=sigma)

    # Sobel edge magnitude
    edges = skfilters.sobel(gray)

    # Threshold: pixels above the mean gradient
    threshold = np.mean(edges)
    edge_mask = edges > threshold  # bool (H, W)

    # Per-pixel ΔE
    lab_true = _to_float_lab(img_true)
    lab_pred = _to_float_lab(img_pred)
    delta_e = skcolor.deltaE_ciede2000(lab_true, lab_pred)

    edge_pixels = delta_e[edge_mask]
    if edge_pixels.size == 0:
        return 0.0
    return float(np.mean(edge_pixels))


# ---------------------------------------------------------------------------
# Colorfulness
# ---------------------------------------------------------------------------

def compute_colorfulness(img_rgb: np.ndarray) -> float:
    """Compute the colourfulness metric of Hasler & Süsstrunk (2003).

    The metric is defined as::

        C = σ_rgyb + 0.3 · μ_rgyb

    where ``rg = R − G`` and ``yb = 0.5·(R + G) − B``, and σ / μ are the
    standard deviation and mean of ``sqrt(rg² + yb²)``.

    Parameters
    ----------
    img_rgb:
        Input image, shape ``(H, W, 3)``, dtype uint8 (or any numeric type).

    Returns
    -------
    float
        Colourfulness score ≥ 0.  Gray images score near 0.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must have shape (H, W, 3), got {img_rgb.shape}")

    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    rgyb = np.sqrt(rg ** 2 + yb ** 2)
    sigma = float(np.std(rgyb))
    mu = float(np.mean(rgyb))
    return sigma + 0.3 * mu


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def compute_lpips(
    img_true: np.ndarray,
    img_pred: np.ndarray,
    device: str = "cpu",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Requires the ``lpips`` package and ``torch``.  Returns ``float('nan')``
    gracefully if either is unavailable.

    Parameters
    ----------
    img_true:
        Ground-truth image, shape ``(H, W, 3)``, dtype uint8 RGB.
    img_pred:
        Predicted/colorized image, same shape and dtype.
    device:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    float
        LPIPS distance (lower is more similar).  Returns NaN if the required
        libraries are not installed.
    """
    _check_uint8_rgb(img_true, "img_true")
    _check_uint8_rgb(img_pred, "img_pred")

    try:
        import torch  # type: ignore[import]
        import lpips as _lpips  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "lpips/torch not installed; compute_lpips returns NaN. "
            "Install with: pip install lpips torch",
            stacklevel=2,
        )
        return float("nan")

    cache_key = str(device)
    loss_fn = _LPIPS_MODELS.get(cache_key)
    if loss_fn is None:
        loss_fn = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        _LPIPS_MODELS[cache_key] = loss_fn

    def _to_tensor(img: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 127.5 - 1.0  # [-1,1]
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        dist = loss_fn(_to_tensor(img_true), _to_tensor(img_pred))

    return float(dist.item())


def compute_inception_features(img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Extract 2048-D Inception-v3 pool features for one uint8 RGB image.

    Uses the ImageNet-pretrained torchvision Inception-v3 classifier with its
    final classification layer replaced by an identity. Images are resized to
    299x299 and normalized to [-1, 1]. The pretrained weights are downloaded
    by torchvision on first use if they are not cached locally.
    """
    _check_uint8_rgb(img_rgb)
    try:
        import torch
        import torch.nn.functional as torch_f
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as exc:
        raise ImportError(
            "torch and torchvision are required for E3 FID feature extraction"
        ) from exc

    cache_key = str(device)
    model = _INCEPTION_MODELS.get(cache_key)
    if model is None:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.to(device).eval()
        _INCEPTION_MODELS[cache_key] = model

    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    tensor = tensor.to(device) / 127.5 - 1.0
    tensor = torch_f.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
    with torch.no_grad():
        features = model(tensor)
    if hasattr(features, "logits"):
        features = features.logits
    return features.reshape(features.shape[0], -1)[0].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
) -> float:
    """Compute the Fréchet Inception Distance (FID) from pre-extracted features.

    Parameters
    ----------
    real_feats:
        Feature matrix for real images, shape ``(N, D)`` float32/float64.
    fake_feats:
        Feature matrix for generated images, same shape.

    Returns
    -------
    float
        FID score (lower is better).  Returns ``float('nan')`` if scipy is
        not installed (required for matrix square root).

    Notes
    -----
    The Fréchet distance between two Gaussians is::

        FID = ||μ₁ − μ₂||² + Tr(Σ₁ + Σ₂ − 2·(Σ₁Σ₂)^½)

    Implementation follows Heusel et al. (2017).
    """
    if not _HAS_SCIPY:
        warnings.warn(
            "scipy is required for FID (matrix square root). "
            "Install it with: pip install scipy",
            stacklevel=2,
        )
        return float("nan")

    real_feats = np.asarray(real_feats, dtype=np.float64)
    fake_feats = np.asarray(fake_feats, dtype=np.float64)

    if real_feats.ndim != 2 or fake_feats.ndim != 2:
        raise ValueError("real_feats and fake_feats must be 2-D arrays (N, D).")
    if real_feats.shape[1] != fake_feats.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: real={real_feats.shape[1]}, "
            f"fake={fake_feats.shape[1]}"
        )

    # For the common small-set/high-dimensional case (e.g. 20 COCO images and
    # 2048 Inception features), evaluate the covariance cross-term in sample
    # space. This is algebraically equivalent and avoids a 2048x2048 matrix
    # square root for every domain/model pair.
    if real_feats.shape[0] < 2 or fake_feats.shape[0] < 2:
        raise ValueError("FID requires at least two real and two generated feature rows")
    if real_feats.shape[1] > min(real_feats.shape[0], fake_feats.shape[0]):
        mu_r = np.mean(real_feats, axis=0)
        mu_f = np.mean(fake_feats, axis=0)
        centered_r = real_feats - mu_r
        centered_f = fake_feats - mu_f
        cov_trace = (
            float(np.sum(centered_r ** 2) / (real_feats.shape[0] - 1))
            + float(np.sum(centered_f ** 2) / (fake_feats.shape[0] - 1))
        )
        cross = centered_r @ centered_f.T
        cross /= math.sqrt((real_feats.shape[0] - 1) * (fake_feats.shape[0] - 1))
        cross_trace = float(np.sum(np.linalg.svd(cross, compute_uv=False)))
        mean_term = float((mu_r - mu_f) @ (mu_r - mu_f))
        return max(0.0, mean_term + cov_trace - 2.0 * cross_trace)

    mu_r = np.mean(real_feats, axis=0)
    mu_f = np.mean(fake_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f

    # Matrix square root via scipy.
    # Note: in newer scipy versions sqrtm() returns only the matrix (no disp kwarg).
    cov_product = sigma_r @ sigma_f
    try:
        sqrt_cov_result = splinalg.sqrtm(cov_product, disp=False)
        sqrt_cov = sqrt_cov_result[0] if isinstance(sqrt_cov_result, tuple) else sqrt_cov_result
    except TypeError:
        sqrt_cov = splinalg.sqrtm(cov_product)

    # Handle numerical issues (small imaginary parts)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = np.real(sqrt_cov)

    fid = (
        float(diff @ diff)
        + float(np.trace(sigma_r))
        + float(np.trace(sigma_f))
        - 2.0 * float(np.trace(sqrt_cov))
    )
    return float(fid)


# ---------------------------------------------------------------------------
# mIoU
# ---------------------------------------------------------------------------

def compute_miou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
) -> float:
    """Compute mean Intersection-over-Union (mIoU) for semantic segmentation.

    Parameters
    ----------
    pred_mask:
        Predicted class labels, integer array of shape ``(H, W)``.
    gt_mask:
        Ground-truth class labels, same shape.
    num_classes:
        Total number of classes (labels range 0 … num_classes−1).

    Returns
    -------
    float
        mIoU in ``[0, 1]``.  Classes absent from both masks are excluded from
        the mean.
    """
    pred_mask = np.asarray(pred_mask, dtype=np.int64)
    gt_mask = np.asarray(gt_mask, dtype=np.int64)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask={pred_mask.shape}, gt_mask={gt_mask.shape}"
        )

    iou_list: List[float] = []
    for cls in range(num_classes):
        pred_c = pred_mask == cls
        gt_c = gt_mask == cls
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue  # Class absent from both → skip
        iou_list.append(float(intersection) / float(union))

    if not iou_list:
        return 0.0
    return float(np.mean(iou_list))


# ---------------------------------------------------------------------------
# Pixel coverage
# ---------------------------------------------------------------------------

def compute_pixel_coverage(decomposition: "Decomposition") -> float:  # type: ignore[name-defined]
    """Return the fraction of pixels owned by a named region (not fallback).

    This is equivalent to ``decomposition.coverage``, but is re-computed
    directly from ``decomposition.owner`` for transparency.

    Parameters
    ----------
    decomposition:
        A :class:`colorize.regions.Decomposition` instance produced by the
        routing stage.

    Returns
    -------
    float
        Fraction of pixels in ``[0, 1]`` that are assigned to some region
        (``owner >= 0``).
    """
    owner = np.asarray(decomposition.owner)
    total_pixels = owner.size
    if total_pixels == 0:
        return 0.0
    covered = int(np.sum(owner >= 0))
    return covered / total_pixels


# ---------------------------------------------------------------------------
# Routing accuracy
# ---------------------------------------------------------------------------

def compute_routing_accuracy(
    regions: Sequence,  # Sequence[Region]
    gt_domain_map: np.ndarray,
    domain_order: Sequence[str],
) -> float:
    """Compute the fraction of region pixels where the predicted domain matches GT.

    Parameters
    ----------
    regions:
        List of :class:`colorize.regions.Region` objects, each with a
        ``domain`` attribute (str) and a ``mask`` attribute (bool H×W array).
    gt_domain_map:
        Integer array of shape ``(H, W)``; each value is an index into
        *domain_order* giving the ground-truth domain for that pixel.
    domain_order:
        Sequence of domain names; ``gt_domain_map[y, x] == i`` means the GT
        domain is ``domain_order[i]``.

    Returns
    -------
    float
        Routing accuracy in ``[0, 1]``.  Returns 0.0 if no regions are given.
    """
    gt_domain_map = np.asarray(gt_domain_map, dtype=np.int64)
    domain_order = list(domain_order)

    total_pixels = 0
    correct_pixels = 0

    for region in regions:
        if region.domain is None:
            continue
        mask = np.asarray(region.mask, dtype=bool)
        if mask.sum() == 0:
            continue

        pred_idx = domain_order.index(region.domain) if region.domain in domain_order else -1
        if pred_idx < 0:
            total_pixels += int(mask.sum())
            continue

        gt_idx_pixels = gt_domain_map[mask]
        correct = int(np.sum(gt_idx_pixels == pred_idx))
        correct_pixels += correct
        total_pixels += int(mask.sum())

    if total_pixels == 0:
        return 0.0
    return correct_pixels / total_pixels


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    regions: Sequence,  # Sequence[Region]
    gt_domain_map: np.ndarray,
    domain_order: Sequence[str],
) -> np.ndarray:
    """Compute a pixel-level routing confusion matrix.

    Entry ``cm[i, j]`` counts how many pixels belong to GT domain *i* but
    were predicted as domain *j*.

    Parameters
    ----------
    regions:
        List of :class:`colorize.regions.Region` objects with ``domain`` and
        ``mask`` attributes.
    gt_domain_map:
        Integer array of shape ``(H, W)``; values are indices into
        *domain_order*.
    domain_order:
        Ordered list of domain names.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape ``(D, D)`` where D = len(domain_order),
        dtype int64.
    """
    gt_domain_map = np.asarray(gt_domain_map, dtype=np.int64)
    domain_order = list(domain_order)
    D = len(domain_order)
    cm = np.zeros((D, D), dtype=np.int64)

    for region in regions:
        if region.domain is None:
            continue
        mask = np.asarray(region.mask, dtype=bool)
        if mask.sum() == 0:
            continue

        pred_idx = domain_order.index(region.domain) if region.domain in domain_order else -1
        if pred_idx < 0:
            continue

        gt_idx_pixels = gt_domain_map[mask]
        for gt_idx in range(D):
            count = int(np.sum(gt_idx_pixels == gt_idx))
            if count > 0:
                cm[gt_idx, pred_idx] += count

    return cm


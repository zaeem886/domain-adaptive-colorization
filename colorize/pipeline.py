"""Top-level colorization pipeline.

Ties together:
  RegionExtractor → ExpertManager → Blender → Reconstructor

Typical usage::

    from colorize.pipeline import ColorizationPipeline, ColorizationConfig
    from colorize.experts import ExpertRegistry

    registry: ExpertRegistry = {
        "people":  "checkpoints/ddcolor_people.pth",
        "nature":  "checkpoints/ddcolor_nature.pth",
        "urban":   "checkpoints/ddcolor_urban.pth",
        "animals": "checkpoints/ddcolor_animals.pth",
        "fallback": "checkpoints/ddcolor_general.pth",
    }

    pipeline = ColorizationPipeline(registry, cfg=ColorizationConfig(margin=0.2))
    rgb = pipeline.run(L_channel, decomposition)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .blending import Blender, Reconstructor
from .colorization_config import ColorizationConfig
from .experts import ExpertManager, ExpertRegistry
from .extraction import CropBox, RegionExtractor
from .regions import Decomposition, Region


@dataclass
class RegionTrace:
    """Intermediate data collected for one region (fallback included)."""

    index: int
    region: Region
    crop: CropBox
    L_crop: np.ndarray        # (S, S) expert input
    ab_pred: np.ndarray       # (S, S, 2) raw expert output
    ab_canvas: np.ndarray     # (H, W, 2) mapped back to the working image
    binary_mask: np.ndarray   # (H, W) mask the blender used
    owner_mask: np.ndarray    # (H, W) disjoint resolved mask
    is_fallback: bool = False


@dataclass
class PipelineTrace:
    """Everything needed to render the intermediate pipeline stages."""

    original_shape: Tuple[int, int]
    working_shape: Tuple[int, int]
    L_full: np.ndarray        # (H, W) working-resolution luminance
    regions: List[RegionTrace]
    ab_blended: np.ndarray    # (H, W, 2)
    rgb_final: np.ndarray     # (H_orig, W_orig, 3)


# manages the entire colorization pipeline
class ColorizationPipeline:
    """End-to-end MoE colorization pipeline (Experiment E4).

    Parameters
    ----------
    registry:
        Maps domain names → checkpoint paths.
    cfg:
        Colorization hyper-parameters (margin, smoothing_width, …).
    device:
        PyTorch device string forwarded to the expert models.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        cfg: Optional[ColorizationConfig] = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg or ColorizationConfig()
        self.registry = registry
        self.device = device

    def run(
        self,
        L_full: np.ndarray,
        decomp: Decomposition,
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Run the full pipeline and return an RGB image.

        Parameters
        ----------
        L_full:
            L channel at working resolution, shape ``(H, W)``,
            float32, range [0, 100].
        decomp:
            :class:`~colorize.regions.Decomposition` produced by Person A's
            routing stage.
        original_shape:
            ``(H_orig, W_orig)`` before pipeline downscaling.  Defaults to
            the shape of *L_full* (no upscaling needed).

        Returns
        -------
        rgb:
            ``(H_orig, W_orig, 3)`` uint8 RGB.
        """

        # wow just calls another function
        rgb, _ = self._run(L_full, decomp, original_shape, collect_trace=False)
        return rgb

    def run_traced(
        self,
        L_full: np.ndarray,
        decomp: Decomposition,
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, PipelineTrace]:
        """Run the pipeline and also return the intermediate per-region results.

        Same inputs and output image as :meth:`run`; the second return value
        holds the per-region crops, expert predictions, masks, and blended
        chrominance so they can be rendered afterwards.
        """
        rgb, trace = self._run(L_full, decomp, original_shape, collect_trace=True)
        assert trace is not None
        return rgb, trace

    # this is the ACTUAL run function
    def _run(
        self,
        L_full: np.ndarray,
        decomp: Decomposition,
        original_shape: Optional[Tuple[int, int]],
        collect_trace: bool,
    ) -> Tuple[np.ndarray, Optional[PipelineTrace]]:
        H, W = L_full.shape[:2]
        working_shape = (H, W)
        orig_shape = original_shape or working_shape

        # initialize the extractor and blender classes
        extractor = RegionExtractor(self.cfg, working_shape)
        blender = Blender(self.cfg, working_shape)

        # Collect all regions: routed regions + fallback
        # The fallback mask covers all uncovered pixels (owner == -1).
        fallback_mask = decomp.owner < 0
        fallback_box = _mask_to_box(fallback_mask, H, W)
        fallback_region = Region(
            label="fallback",
            score=1.0,
            box=np.array(fallback_box, dtype=float),
            mask=fallback_mask,
            domain=decomp.domains[-1],   # last domain is always the fallback
        )

        all_regions = decomp.regions + [fallback_region]

        ab_canvases = []
        binary_masks = []
        crops = []
        trace_regions: List[RegionTrace] = []

        # route each region to its own expert
        with ExpertManager(
            self.registry,
            self.device,
            input_size=self.cfg.input_size,
        ) as manager:
            for index, region in enumerate(all_regions):
                domain = region.domain
                L_crop, crop = extractor.extract_input(L_full, region)

                # Colorize
                ab_pred = manager.colorize(domain, L_crop)  # (S, S, 2)

                # Map back to full canvas
                ab_canvas = extractor.map_output(ab_pred, crop)
                bin_mask = extractor.get_binary_mask(region, crop)

                ab_canvases.append(ab_canvas)
                binary_masks.append(bin_mask)
                crops.append(crop)

                # more tracing stuff (just collecting the intermediate resultks)
                if collect_trace:
                    is_fallback = index >= len(decomp.regions)
                    owner_mask = (
                        fallback_mask if is_fallback else decomp.resolved_mask(index)
                    )
                    trace_regions.append(
                        RegionTrace(
                            index=index,
                            region=region,
                            crop=crop,
                            L_crop=L_crop,
                            ab_pred=ab_pred,
                            ab_canvas=ab_canvas,
                            binary_mask=bin_mask,
                            owner_mask=owner_mask,
                            is_fallback=is_fallback,
                        )
                    )

        # Blend
        ab_blended = blender.blend(ab_canvases, binary_masks, crops)

        # Reconstruct
        recon = Reconstructor(orig_shape, working_shape)
        rgb = recon.reconstruct(
            # L0 must be at original resolution; upsample if needed.
            _upsample_L(L_full, orig_shape),
            ab_blended,
        )

        trace = None
        if collect_trace:
            trace = PipelineTrace(
                original_shape=orig_shape,
                working_shape=working_shape,
                L_full=L_full,
                regions=trace_regions,
                ab_blended=ab_blended,
                rgb_final=rgb,
            )
        return rgb, trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_to_box(mask: np.ndarray, H: int, W: int) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) bounding box of a boolean mask.

    Falls back to the full image if the mask is empty.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0, 0, W, H)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _upsample_L(L: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bicubic-upsample L to target_shape if necessary."""
    import cv2
    H_t, W_t = target_shape
    if L.shape[:2] == (H_t, W_t):
        return L
    return cv2.resize(L, (W_t, H_t), interpolation=cv2.INTER_CUBIC)

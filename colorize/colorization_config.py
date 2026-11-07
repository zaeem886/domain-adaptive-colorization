"""Configuration for the colorization pipeline (Experiment E4).

This is separate from DecompConfig (decomposition / routing parameters) so that
ablations over margin, smoothing_width, and blending_on can be run independently
of the detector/SAM settings.
"""
from dataclasses import dataclass


@dataclass
class ColorizationConfig:
    """Hyper-parameters for region extraction, expert routing, and blending.

    Attributes
    ----------
    margin:
        Context margin ρ added around each region bounding box.  A value of 0.2
        means the crop is expanded by 20 % of the box width on each horizontal
        side and 20 % of the box height on each vertical side.
    smoothing_width:
        Gaussian standard deviation σ₀ (pixels) used to build soft blend masks.
        The per-region σ_k is capped at this value.
    blending_on:
        When True  → soft Gaussian-blended transition between regions.
        When False → hard binary-mask paste (no Gaussian smoothing).
    region_shape:
        ``"mask"``  — the binary mask B_k is taken directly from the SAM output.
        ``"box"``   — the binary mask B_k is the filled bounding box of the region.
    input_size:
        Side length S (pixels) to which every region crop is resized before it is
        passed to the expert model.  Experts are trained at this resolution.
    """

    margin: float = 0.2
    smoothing_width: float = 10.0
    blending_on: bool = True
    region_shape: str = "mask"   # "mask" | "box"
    input_size: int = 256

    def __post_init__(self) -> None:
        if self.region_shape not in ("mask", "box"):
            raise ValueError(f"region_shape must be 'mask' or 'box', got {self.region_shape!r}")
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.smoothing_width < 0:
            raise ValueError(f"smoothing_width must be non-negative, got {self.smoothing_width}")
        if self.input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {self.input_size}")

"""Shared region record: the contract between decomposition, routing and extraction."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Region:
    label: str                 # text label returned by the detector
    score: float               # detector confidence
    box: np.ndarray            # (4,) x1, y1, x2, y2 in pixels
    mask: np.ndarray           # (H, W) bool, the raw SAM mask
    domain: Optional[str] = None   # set by routing

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Decomposition:
    shape: Tuple[int, int]             # (H, W)
    regions: List[Region]
    domains: Tuple[str, ...]           # domain_map index order; last is the fallback
    owner: np.ndarray                  # (H, W) int32: region index owning the pixel, -1 if none
    domain_map: np.ndarray             # (H, W) int8: index into `domains`, never -1
    coverage: float                    # fraction of pixels owned by some region
    domain_share: Dict[str, float]     # fraction of pixels per domain (sums to 1)
    discarded: List[Region] = field(default_factory=list)  # regions whose label could not be routed

    def domain_mask(self, domain: str) -> np.ndarray:
        """Boolean mask of the pixels assigned to a domain (after overlap resolution)."""
        return self.domain_map == self.domains.index(domain)

    def resolved_mask(self, region_index: int) -> np.ndarray:
        """Pixels a region owns after overlap resolution (disjoint across regions)."""
        return self.owner == region_index


def save_decomposition(d: Decomposition, path: str) -> None:
    """Save a decomposition so later stages and experiments can reload it."""
    boxes = np.stack([r.box for r in d.regions]) if d.regions else np.zeros((0, 4))
    np.savez_compressed(
        path,
        owner=d.owner,
        domain_map=d.domain_map,
        domains=np.array(d.domains, dtype=str),
        labels=np.array([r.label for r in d.regions], dtype=str),
        region_domains=np.array([r.domain for r in d.regions], dtype=str),
        scores=np.array([r.score for r in d.regions], dtype=float),
        boxes=boxes,
    )

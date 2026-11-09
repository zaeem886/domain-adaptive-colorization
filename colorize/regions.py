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
    """Save a decomposition so later stages and experiments can reload it.

    The routed regions are stored by metadata plus the ``owner`` map.  Raw SAM
    masks are deliberately not persisted because they can be large; this is
    enough for evaluation, which uses the resolved masks (owner == region
    index).  Discarded regions are stored as a count and their labels so E1/E2
    can report how many detections routing could not place.
    """
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
        n_discarded=np.array(len(d.discarded), dtype=np.int32),
        discarded_labels=np.array([r.label for r in d.discarded], dtype=str),
    )


def load_decomposition(path: str) -> Decomposition:
    """Load a decomposition written by :func:`save_decomposition`.

    Only the routed region metadata and the owner map are available on disk.
    Each routed region therefore gets ``mask = (owner == index)``, i.e. its
    resolved mask, not the original raw SAM mask.  E2 relies on this resolved
    mask; callers that need raw masks must re-run SAM.

    Discarded regions are reconstructed with their label only: their boxes and
    scores are not persisted, and their masks are empty.  This is sufficient
    for the counts and labels reported by E1/E2.
    """
    with np.load(path, allow_pickle=False) as data:
        owner = np.asarray(data["owner"])
        shape = tuple(int(v) for v in owner.shape)
        domains = tuple(str(v) for v in data["domains"].tolist())
        domain_map = np.asarray(data["domain_map"])

        labels = [str(v) for v in data["labels"].tolist()] if "labels" in data.files else []
        region_domains = (
            [str(v) for v in data["region_domains"].tolist()]
            if "region_domains" in data.files
            else [None] * len(labels)
        )
        scores = (
            np.asarray(data["scores"], dtype=float)
            if "scores" in data.files
            else np.zeros(len(labels), dtype=float)
        )
        boxes = (
            np.asarray(data["boxes"], dtype=float)
            if "boxes" in data.files
            else np.zeros((len(labels), 4), dtype=float)
        )
        n_discarded = int(np.asarray(data["n_discarded"]).item()) if "n_discarded" in data.files else 0
        discarded_labels = (
            [str(v) for v in data["discarded_labels"].tolist()]
            if "discarded_labels" in data.files
            else []
        )

    regions = [
        Region(
            label=labels[i],
            score=float(scores[i]),
            box=np.asarray(boxes[i], dtype=float),
            mask=(owner == i),
            domain=None if region_domains[i] == "None" else region_domains[i],
        )
        for i in range(len(labels))
    ]
    discarded = [
        Region(
            label=discarded_labels[i],
            score=float("nan"),
            box=np.zeros(4, dtype=float),
            mask=np.zeros(shape, dtype=bool),
        )
        for i in range(min(n_discarded, len(discarded_labels)))
    ]

    covered = int((owner >= 0).sum())
    total = int(owner.size)
    domain_share = {
        name: float((domain_map == index).sum()) / total if total else float("nan")
        for index, name in enumerate(domains)
    }
    return Decomposition(
        shape=shape,
        regions=regions,
        domains=domains,
        owner=owner,
        domain_map=domain_map,
        coverage=float(covered / total) if total else float("nan"),
        domain_share=domain_share,
        discarded=discarded,
    )

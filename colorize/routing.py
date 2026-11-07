"""Routing: label -> domain, overlap resolution, fallback for uncovered pixels.
Pure numpy so it can be unit-tested without a GPU or any model."""
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .config import DecompConfig
from .regions import Decomposition, Region


def build_label_table(prompts: Dict[str, List[str]]) -> Dict[str, str]:
    """Flatten {domain: [phrases]} into {phrase: domain}."""
    table: Dict[str, str] = {}
    for domain, phrases in prompts.items():
        for p in phrases:
            table.setdefault(p.lower().strip(), domain)
    return table


def label_to_domain(label: str, table: Dict[str, str], split_merged: bool = False) -> Optional[str]:
    """Map a detector label to a domain, or None if it cannot be routed.

    Report section 5.3.1: a label that does not match exactly one prompt is not routed.
    With split_merged=True, a merged label such as "person face" is split by phrase
    matching and the most common domain wins (ties: first match in the table).
    """
    label = label.lower().strip()
    if label in table:
        return table[label]
    if not split_merged:
        return None
    votes: Counter = Counter()
    for phrase, domain in table.items():
        if re.search(rf"\b{re.escape(phrase)}\b", label):
            votes[domain] += 1
    return votes.most_common(1)[0][0] if votes else None


def resolve_overlaps(regions: Sequence[Region], shape: Tuple[int, int], rule: str = "smaller") -> np.ndarray:
    """Return an (H, W) int32 map of the region owning each pixel (-1 = none).

    Regions are painted in an order that lets the winner be painted last.
    """
    owner = np.full(shape, -1, dtype=np.int32)
    areas = [r.area for r in regions]
    idx = range(len(regions))
    if rule == "smaller":
        order = sorted(idx, key=lambda i: (-areas[i], regions[i].score))
    elif rule == "score":
        order = sorted(idx, key=lambda i: (regions[i].score, -areas[i]))
    else:
        raise ValueError(f"unknown overlap rule: {rule!r}")
    for i in order:
        owner[regions[i].mask] = i
    return owner


def route(regions: List[Region], shape: Tuple[int, int], cfg: Optional[DecompConfig] = None) -> Decomposition:
    """Assign domains, discard unroutable regions, resolve overlaps, apply the fallback.

    Unroutable regions are discarded before overlap resolution, so their pixels count
    as uncovered and go to the fallback (section 5.3.1 and 5.3.3).
    """
    cfg = cfg or DecompConfig()
    for r in regions:
        if r.mask.shape != tuple(shape):
            raise ValueError(f"mask shape {r.mask.shape} does not match image shape {tuple(shape)}")

    table = build_label_table(cfg.prompts)
    routed: List[Region] = []
    discarded: List[Region] = []
    for r in regions:
        domain = label_to_domain(r.label, table, cfg.split_merged_labels)
        if domain is None:
            discarded.append(r)
        else:
            r.domain = domain
            routed.append(r)

    domains = cfg.domain_order
    fb_idx = domains.index(cfg.fallback)
    owner = resolve_overlaps(routed, shape, cfg.overlap_rule)

    domain_map = np.full(shape, fb_idx, dtype=np.int8)
    for i, r in enumerate(routed):
        domain_map[owner == i] = domains.index(r.domain)

    coverage = float((owner >= 0).mean())
    total = domain_map.size
    share = {d: float((domain_map == k).sum()) / total for k, d in enumerate(domains)}
    return Decomposition(tuple(shape), routed, domains, owner, domain_map, coverage, share, discarded)

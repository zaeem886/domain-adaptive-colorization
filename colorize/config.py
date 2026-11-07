"""Settings for decomposition and routing. Nothing here imports torch."""
from dataclasses import dataclass, field
from typing import Dict, List

DOMAINS = ("people", "nature", "urban", "animals")
FALLBACK = "fallback"

# TODO: replace with the exact prompt set from Table 5.1 of the report.
DEFAULT_PROMPTS: Dict[str, List[str]] = {
    "people": ["person", "face", "man", "woman", "child"],
    "nature": ["sky", "grass", "tree", "mountain", "water", "sand", "cloud", "field"],
    "urban": ["building", "road", "car", "bridge", "house", "street", "bus", "truck"],
    "animals": ["dog", "cat", "bird", "horse", "cow", "sheep", "elephant"],
}


@dataclass
class DecompConfig:
    prompts: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROMPTS.items()}
    )
    # Detection thresholds (swept in experiment E1).
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    # Drop regions smaller than this fraction of the image.
    min_area_frac: float = 0.002
    # Cap on boxes passed to SAM, highest scores kept (memory guard).
    max_boxes: int = 40
    # How pixels claimed by several regions are resolved:
    #   "smaller": the smaller region wins (ties: higher score). Small objects
    #              such as a dog on grass keep their own pixels.
    #   "score":   the higher-scoring region wins (ties: smaller region).
    # TODO: align with the overlap rule in section 5.3.2 of the report.
    overlap_rule: str = "smaller"
    fallback: str = FALLBACK
    dino_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_id: str = "facebook/sam-vit-base"

    @property
    def domain_order(self):
        """Index order used by Decomposition.domain_map; fallback is last."""
        return tuple(self.prompts.keys()) + (self.fallback,)

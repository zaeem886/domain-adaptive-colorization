"""Settings for decomposition and routing. Nothing here imports torch.

Defaults follow the report: prompt set = Table 5.1, label mapping = Table 5.2,
overlap rule = section 5.3.2 (smallest region wins, ties by higher score).
Threshold values are the ones used in the earlier Grounded-SAM notebook; the
report says the six parameters are tuned on a tuning split and then fixed.
"""
from dataclasses import dataclass, field
from typing import Dict, List

DOMAINS = ("people", "nature", "urban", "animals")
FALLBACK = "fallback"

# Table 5.1 / Table 5.2. Each prompt belongs to exactly one domain.
# The old notebook also used "car"; it is not in Table 5.2, so it is left out.
# To add it, put "car" under "urban" here AND update the table in the report.
DEFAULT_PROMPTS: Dict[str, List[str]] = {
    "people": ["person", "face"],
    "nature": ["sky", "vegetation", "tree"],
    "urban": ["building"],
    "animals": ["animal"],
}


@dataclass
class DecompConfig:
    prompts: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROMPTS.items()}
    )
    # The six decomposition parameters from section 5.2.3.
    max_side: int = 1024            # longer image side after Lanczos downscaling
    box_threshold: float = 0.35     # Grounding DINO box confidence
    text_threshold: float = 0.25    # Grounding DINO text match
    min_score: float = 0.35         # minimum detection score kept
    nms_iou: float = 0.5            # per-class NMS suppression threshold
    merge_containment: float = 0.6  # same-label fragment merge threshold

    # Routing (section 5.3).
    overlap_rule: str = "smaller"   # "smaller" (report) or "score" (ablation only)
    # The report discards a label that does not match exactly one prompt.
    # Set True to split merged labels such as "person face" by phrase matching.
    split_merged_labels: bool = False
    fallback: str = FALLBACK

    # Models.
    dino_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_type: str = "vit_b"
    sam_checkpoint: str = "sam_vit_b.pth"
    sam_url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    @property
    def domain_order(self):
        """Index order used by Decomposition.domain_map; fallback is last."""
        return tuple(self.prompts.keys()) + (self.fallback,)

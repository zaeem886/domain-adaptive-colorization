"""Overlay a decomposition on the image (for slides and the report)."""
import numpy as np
from PIL import Image, ImageDraw

from .regions import Decomposition

DOMAIN_COLORS = {
    "people": (229, 86, 138), "nature": (93, 178, 107),
    "urban": (76, 141, 240), "animals": (242, 177, 52), "fallback": (142, 142, 147),
}


def overlay(image: Image.Image, d: Decomposition, alpha: float = 0.45, boxes: bool = True) -> Image.Image:
    base = np.asarray(image.convert("RGB")).astype(float)
    layer = base.copy()
    for k, name in enumerate(d.domains):
        layer[d.domain_map == k] = DOMAIN_COLORS.get(name, (200, 200, 200))
    out = Image.fromarray(((1 - alpha) * base + alpha * layer).astype(np.uint8))
    if boxes:
        draw = ImageDraw.Draw(out)
        for r in d.regions:
            color = DOMAIN_COLORS.get(r.domain, (255, 255, 255))
            draw.rectangle(list(map(float, r.box)), outline=color, width=2)
            draw.text((r.box[0] + 3, r.box[1] + 2), r.label, fill=(255, 255, 255))
    return out

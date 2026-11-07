"""Region decomposition (report section 5.2.3): grayscale load and resize, one DINO
pass, filtering, SAM box prompts, fragment merge, then routing.

Grounding DINO comes from transformers; SAM from the original `segment_anything`
package (the version tested in the earlier notebook). Models load lazily on first use.
"""
import inspect
import os
import urllib.request
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .config import DecompConfig
from .filtering import filter_detections, merge_same_label
from .regions import Decomposition, Region
from .routing import route


def load_gray(source: Union[str, Image.Image], max_side: int = 1024) -> Image.Image:
    """Step 1: read as single-channel grayscale; if the longer side exceeds max_side,
    downscale with Lanczos keeping the aspect ratio. All masks live at this resolution."""
    img = source.convert("L") if isinstance(source, Image.Image) else Image.open(source).convert("L")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


class GroundedSAM:
    def __init__(self, cfg: Optional[DecompConfig] = None, device: Optional[str] = None):
        self.cfg = cfg or DecompConfig()
        self._device = device
        self._loaded = False

    # ---- model loading ----------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from segment_anything import SamPredictor, sam_model_registry

        self.torch = torch
        self.device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_proc = AutoProcessor.from_pretrained(self.cfg.dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.dino_id).to(self.device).eval()

        if not os.path.exists(self.cfg.sam_checkpoint):
            urllib.request.urlretrieve(self.cfg.sam_url, self.cfg.sam_checkpoint)
        sam = sam_model_registry[self.cfg.sam_type](checkpoint=self.cfg.sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self._loaded = True

    @property
    def caption(self) -> str:
        """Step 2: one lowercase string, entries separated by a period, ending in a period."""
        phrases = [p.lower().strip() for group in self.cfg.prompts.values() for p in group]
        return " . ".join(dict.fromkeys(phrases)) + " ."

    # ---- the two model calls (overridden in tests) ------------------------
    def _run_dino(self, rgb: Image.Image) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Step 3: boxes (N,4 xyxy pixels), scores (N,), labels (N strings)."""
        self._load()
        torch = self.torch
        inputs = self.dino_proc(images=rgb, text=self.caption, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino(**inputs)
        fn = self.dino_proc.post_process_grounded_object_detection
        kw = dict(text_threshold=self.cfg.text_threshold, target_sizes=[rgb.size[::-1]])
        # The argument name changed across transformers versions (box_threshold -> threshold).
        kw["threshold" if "threshold" in inspect.signature(fn).parameters else "box_threshold"] = self.cfg.box_threshold
        res = fn(outputs, inputs.input_ids, **kw)[0]
        raw_labels = res["text_labels"] if "text_labels" in res else res["labels"]
        labels = [str(x) for x in raw_labels]
        return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels

    def _run_sam(self, rgb_array: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """Step 5: encode the image once, then one box prompt per region."""
        self._load()
        self.sam_predictor.set_image(rgb_array)
        masks = []
        for box in boxes:
            m, _, _ = self.sam_predictor.predict(box=np.array(box), multimask_output=False)
            masks.append(m[0].astype(bool))
        return masks

    # ---- the stage --------------------------------------------------------
    def detect(self, gray: Image.Image) -> List[Region]:
        """Steps 2 to 6 on an already-loaded grayscale image. Domains are set by routing."""
        rgb = gray.convert("RGB")   # three identical channels: the models expect RGB input
        boxes, scores, labels = self._run_dino(rgb)
        if len(boxes) == 0:
            return []
        keep = filter_detections(boxes, scores, labels, self.cfg.min_score, self.cfg.nms_iou)
        if not keep:
            return []
        masks = self._run_sam(np.array(rgb), boxes[keep])
        regions = [Region(label=labels[i], score=float(scores[i]), box=boxes[i].astype(float), mask=m)
                   for i, m in zip(keep, masks)]
        return merge_same_label(regions, self.cfg.merge_containment)

    def __call__(self, image: Union[str, Image.Image]) -> Decomposition:
        gray = load_gray(image, self.cfg.max_side)
        regions = self.detect(gray)
        return route(regions, (gray.height, gray.width), self.cfg)

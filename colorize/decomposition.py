"""Grounded-SAM wrapper (Hugging Face transformers).

Grounding DINO turns the domain text prompts into boxes; SAM turns the boxes
into masks. Models load once in __init__, so reuse one instance for many images.
"""
import inspect
from typing import List, Optional
import numpy as np
from PIL import Image

from .config import DecompConfig
from .regions import Decomposition, Region
from .routing import route


class GroundedSAM:
    def __init__(self, cfg: Optional[DecompConfig] = None, device: Optional[str] = None):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)
        self.cfg = cfg or DecompConfig()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_proc = AutoProcessor.from_pretrained(self.cfg.dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.dino_id).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(self.cfg.sam_id)
        self.sam = SamModel.from_pretrained(self.cfg.sam_id).to(self.device).eval()

    @property
    def text_prompt(self) -> str:
        # Grounding DINO expects lowercase phrases, each ending with a period.
        phrases = [p for group in self.cfg.prompts.values() for p in group]
        return ". ".join(dict.fromkeys(p.lower() for p in phrases)) + "."

    def _post_process(self, outputs, input_ids, size_hw):
        # The argument name changed across transformers versions (box_threshold -> threshold).
        fn = self.dino_proc.post_process_grounded_object_detection
        kw = dict(text_threshold=self.cfg.text_threshold, target_sizes=[size_hw])
        if "threshold" in inspect.signature(fn).parameters:
            kw["threshold"] = self.cfg.box_threshold
        else:
            kw["box_threshold"] = self.cfg.box_threshold
        return fn(outputs, input_ids, **kw)[0]

    def detect(self, image: Image.Image) -> List[Region]:
        """Return raw regions (label, score, box, mask). Domains are set by routing."""
        torch = self.torch
        image = image.convert("RGB")
        w, h = image.size
        with torch.no_grad():
            inputs = self.dino_proc(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
            res = self._post_process(self.dino(**inputs), inputs.input_ids, (h, w))
        labels = res.get("text_labels", res["labels"])
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        if len(boxes) == 0:
            return []

        keep = np.argsort(-scores)[: self.cfg.max_boxes]
        boxes, scores = boxes[keep], scores[keep]
        labels = [str(labels[i]) for i in keep]

        with torch.no_grad():
            sam_in = self.sam_proc(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
            out = self.sam(**sam_in, multimask_output=False)
            masks = self.sam_proc.image_processor.post_process_masks(
                out.pred_masks.cpu(), sam_in["original_sizes"].cpu(), sam_in["reshaped_input_sizes"].cpu()
            )[0]
        masks = masks[:, 0].numpy().astype(bool)   # (N, H, W)

        min_area = self.cfg.min_area_frac * h * w
        return [
            Region(label=lab, score=float(sc), box=bx.astype(float), mask=m)
            for lab, sc, bx, m in zip(labels, scores, boxes, masks)
            if m.sum() >= min_area
        ]

    def __call__(self, image: Image.Image) -> Decomposition:
        regions = self.detect(image)
        w, h = image.size
        return route(regions, (h, w), self.cfg)

"""Expert Colorization: DDColor-T wrapper, registry, and lazy-loading manager.

Architecture reference: DDColor (Kang et al., 2023) with a ConvNeXt-T backbone
(DDColor-T variant).  The model is trained to map a 256×256 L channel to a
256×256 ab chrominance map.

Memory design
-------------
* ``ExpertRegistry``  — a plain dict mapping domain names to checkpoint paths.
  Keeping it separate lets tests supply fake entries without touching disk.
* ``Colorizer``       — a thin nn.Module wrapper; it owns one DDColor-T instance,
  loads the checkpoint once, and exposes a clean ``predict(L) → ab`` interface.
* ``ExpertManager``   — lazy-loads a ``Colorizer`` per domain on first use and
  releases all models after a full image has been processed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Type alias for the checkpoint registry.
ExpertRegistry = Dict[str, str]  # domain → checkpoint path


def default_registry() -> ExpertRegistry:
    """Return an empty registry.  Populate it before running the pipeline."""
    return {}


# ---------------------------------------------------------------------------
# DDColor-T model wrapper
# ---------------------------------------------------------------------------

def _build_ddcolor_t(checkpoint: str, device: torch.device) -> nn.Module:
    """Instantiate DDColor-T (ConvNeXt-T backbone) and load *checkpoint*.

    DDColor is available on PyPI as ``basicsr`` + ``ddcolor``.  We import it
    lazily so the rest of the module can be imported without it being installed
    (useful for unit tests that mock ``_build_ddcolor_t``).

    If the ``ddcolor`` package is not installed, a ``RuntimeError`` is raised
    with an installation hint.
    """
    try:
        from ddcolor import DDColor  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'ddcolor' package is required for expert colorization.  "
            "Install it with:  pip install ddcolor"
        ) from exc

    model = DDColor(
        encoder_name="convnext-t",
        decoder_type="MultiScaleColorDecoder",
        input_size=[256, 256],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("params", ckpt)  # support both raw state-dict and param-wrapped files
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


class Colorizer:
    """Wraps DDColor-T to provide a simple ``predict`` interface.

    Parameters
    ----------
    domain:
        Human-readable domain name (for logging / error messages).
    checkpoint:
        Path to the ``.pth`` checkpoint file.
    device:
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    def __init__(self, domain: str, checkpoint: str, device: str = "cuda") -> None:
        self.domain = domain
        self.checkpoint = checkpoint
        if torch is not None:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the checkpoint into GPU memory (idempotent)."""
        if self._model is not None:
            return
        logger.info("Loading checkpoint for domain '%s' from %s", self.domain, self.checkpoint)
        self._model = _build_ddcolor_t(self.checkpoint, self.device)

    def release(self) -> None:
        """Release GPU memory (idempotent)."""
        if self._model is None:
            return
        logger.info("Releasing expert for domain '%s'", self.domain)
        del self._model
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, L: np.ndarray) -> np.ndarray:
        """Colorize a single L-channel image.

        Parameters
        ----------
        L:
            2-D float32 array of shape ``(S, S)``, range [0, 100]
            (OpenCV CIE-Lab convention).

        Returns
        -------
        ab:
            2-D float32 array of shape ``(S, S, 2)``.
        """
        if self._model is None:
            raise RuntimeError(
                f"Colorizer for domain '{self.domain}' has not been loaded.  "
                "Call load() first."
            )
        # Normalise L to [0, 1] and reshape to (1, 1, S, S)
        L_norm = (L / 100.0).astype(np.float32)
        tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ab_tensor = self._model(tensor)  # (1, 2, S, S)

        ab = ab_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (S, S, 2)
        return ab.astype(np.float32)


# ---------------------------------------------------------------------------
# Expert Manager
# ---------------------------------------------------------------------------

class ExpertManager:
    """Lazily loads and caches one ``Colorizer`` per domain.

    Usage::

        manager = ExpertManager(registry, device="cuda:0")
        ab = manager.colorize("nature", L_patch)
        manager.release_all()   # call after each full image

    Parameters
    ----------
    registry:
        Maps domain names to checkpoint paths.
    device:
        PyTorch device string forwarded to each ``Colorizer``.
    """

    def __init__(self, registry: ExpertRegistry, device: str = "cuda") -> None:
        self._registry = registry
        self._device = device
        self._cache: Dict[str, Colorizer] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, domain: str) -> Colorizer:
        """Return (and lazily load) the expert for *domain*.

        Raises
        ------
        KeyError
            If *domain* is not in the registry.
        """
        if domain not in self._registry:
            raise KeyError(
                f"No checkpoint registered for domain '{domain}'.  "
                f"Available: {list(self._registry)}"
            )
        if domain not in self._cache:
            colorizer = Colorizer(domain, self._registry[domain], self._device)
            colorizer.load()
            self._cache[domain] = colorizer
        return self._cache[domain]

    def colorize(self, domain: str, L: np.ndarray) -> np.ndarray:
        """Convenience wrapper: get the expert and run inference in one call."""
        return self.get(domain).predict(L)

    def release_all(self) -> None:
        """Release every loaded expert from GPU memory.

        Call this once after a full image has been processed to free VRAM
        before the next image.
        """
        for colorizer in self._cache.values():
            colorizer.release()
        self._cache.clear()

    def loaded_domains(self) -> list:
        """Return the list of domains currently loaded in GPU memory."""
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExpertManager":
        return self

    def __exit__(self, *_) -> None:
        self.release_all()

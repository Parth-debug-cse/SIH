"""Vehicle visual Re-ID embeddings — MPS-first, Apple Silicon aware.

Pipeline priority for the embedding backbone:
  1. OSNet / FastReID via ``torchreid`` when installed (best vehicle Re-ID).
  2. Torchvision ResNet trunk (truncated, embedding head) on MPS.
  3. Deterministic spatial-color histogram embedding (no weights needed).

Option 3 exists so tracking + cosine-similarity logic is fully testable and
deployable without downloading weights; options 1-2 kick in automatically
when the libraries are present. All tensor ops use MPS when
``torch.backends.mps.is_available()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def select_reid_device() -> str:
    """Return ``'mps'`` when usable, else ``'cpu'``. Never raises."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row-normalized embedding sets.

    Uses torch on MPS when available (all tensor ops on MPS per directive),
    else numpy. Inputs need not be normalized.
    """
    try:
        import torch

        dev = select_reid_device()
        ta = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        tb = torch.ascontiguousarray(b)
        tb = torch.as_tensor(tb, dtype=torch.float32, device=dev)
        ta = ta / ta.norm(dim=1, keepdim=True).clamp_min(1e-9)
        tb = tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return (ta @ tb.T).to("cpu").numpy()
    except Exception:
        an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
        bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
        return an @ bn.T


def _numpy_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbor resize without OpenCV (degraded-mode helper)."""
    sh, sw = img.shape[:2]
    ys = (np.linspace(0, sh - 1, h)).astype(int)
    xs = (np.linspace(0, sw - 1, w)).astype(int)
    return img[ys][:, xs]


@dataclass
class ReIDConfig:
    embedding_dim: int = 512
    image_size: tuple[int, int] = (128, 256)  # (w, h)
    backbone: str = "auto"  # auto | osnet | resnet | histogram
    model_name: str = "osnet_ain_x1_0"


class VehicleReID:
    """Extracts L2-normalized visual embeddings from vehicle crops.

    The embedding is bound to a Track ID by the MOT tracker (see
    ``src/infrastructure/tracking/mot_tracker.py``): when plate OCR fails,
    identity is maintained purely on cosine similarity of these vectors.
    """

    def __init__(self, config: ReIDConfig | None = None) -> None:
        self.config = config or ReIDConfig()
        self.device = select_reid_device()
        self.backend_name: str = "histogram"
        self._model = None
        self._load_attempted = False

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        want = self.config.backbone
        if want in ("auto", "osnet"):
            if self._try_load_osnet():
                return
        if want in ("auto", "resnet"):
            if self._try_load_resnet():
                return
        self.backend_name = "histogram"
        logger.info("ReID using histogram fallback (dim=%d)", self.config.embedding_dim)

    def _try_load_osnet(self) -> bool:
        try:
            import torch
            from torchreid import models as reid_models

            m = reid_models.build_model(
                name=self.config.model_name, num_classes=1000, pretrained=True
            )
            m.eval().to(self.device)
            self._model = m
            self.backend_name = f"torchreid:{self.config.model_name}"
            logger.info("ReID backbone loaded: %s on %s", self.backend_name, self.device)
            return True
        except Exception as exc:
            logger.debug("torchreid OSNet unavailable: %s", exc)
            return False

    def _try_load_resnet(self) -> bool:
        try:
            import torch
            from torch import nn
            from torchvision import models as tv_models

            trunk = tv_models.resnet18(weights="DEFAULT")
            trunk.fc = nn.Identity()  # 512-D trunk output
            trunk.eval().to(self.device)
            self._model = trunk
            self.backend_name = "resnet18-trunk"
            logger.info("ReID backbone loaded: resnet18-trunk on %s", self.device)
            return True
        except Exception as exc:
            logger.debug("torchvision ResNet unavailable: %s", exc)
            return False

    # ------------------------------------------------------------------
    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of BGR vehicle crops -> (N, D) L2-normalized array."""
        self._ensure_model()
        if self._model is not None:
            try:
                return self._extract_deep(crops)
            except Exception:
                logger.exception("Deep ReID failed; falling back to histogram")
        return self._extract_histogram(crops)

    def extract_one(self, crop: np.ndarray) -> np.ndarray:
        return self.extract([crop])[0]

    # ------------------------------------------------------------------
    def _extract_deep(self, crops: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        w, h = self.config.image_size
        batch = []
        for c in crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            # ImageNet norm (OSNet/ResNet standard)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            batch.append((t - mean) / std)
        x = torch.stack(batch).to(self.device)
        with torch.no_grad():
            feats = self._model(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.flatten(1).float()
            feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return feats.to("cpu").numpy()

    def _extract_histogram(self, crops: list[np.ndarray]) -> np.ndarray:
        """Deterministic spatial-color embedding (weight-free fallback).

        4x4 spatial grid x 8-bin per-channel HSV histogram -> L2-normalized.
        Same vehicle across adjacent frames scores high cosine similarity;
        different-colored vehicles score low — enough to hold a Track ID
        through an OCR outage. Falls back to a numpy-only RGB histogram
        when OpenCV is unavailable (degraded but functional).
        """
        try:
            import cv2

            have_cv2 = True
        except Exception:
            have_cv2 = False

        D = self.config.embedding_dim
        out = np.zeros((len(crops), D), dtype=np.float32)
        for i, crop in enumerate(crops):
            if have_cv2:
                small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                grid = hsv
            else:
                small = _numpy_resize(crop, 64, 64)
                grid = small  # RGB-order raw channels, histogrammed the same way
            cells: list[float] = []
            for gy in range(4):
                for gx in range(4):
                    cell = grid[gy * 16:(gy + 1) * 16, gx * 16:(gx + 1) * 16]
                    for ch in range(3):
                        hist, _ = np.histogram(cell[:, :, ch], bins=8, range=(0, 256))
                        cells.extend((hist / max(cell.size, 1)).tolist())
            # 4*4*3*8 = 384 dims; tile/truncate to D.
            v = np.array(cells, dtype=np.float32)
            reps = int(np.ceil(D / v.size))
            v = np.tile(v, reps)[:D]
            n = float(np.linalg.norm(v))
            out[i] = v / (n if n > 1e-9 else 1.0)
        return out

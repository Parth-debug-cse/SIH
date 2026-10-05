"""LPRNet recognition head compiled for Apple's MLX framework.

LPRNet (Zherzdev & Gruzdev, 2018) is a lightweight fully-convolutional
network: the plate crop goes in, per-column character logits come out, and
greedy CTC decoding yields the string — no segmentation, no RNN, ideal for
the ANE-adjacent MLX path on Apple Silicon.

Weight contract: ``load_weights(path)`` reads a ``.npz`` whose keys are
``<layer-name>`` with values as numpy arrays matching the MLX parameter
shapes (exported once from a trained torch LPRNet). The class NEVER
imports mlx at module scope, so machines without MLX still import cleanly
and the engine can fall back to Vision.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Indian HSRP alphabet: digits + uppercase Latin. CTC blank = index 0.
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0
NUM_CLASSES = len(CHARS) + 1  # blank + 36


def ctc_greedy_decode(logits: np.ndarray) -> tuple[str, float]:
    """Greedy CTC decode of (T, C) log-probs -> (text, mean-char-confidence)."""
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must be (T, {NUM_CLASSES}), got {logits.shape}")
    # Softmax for confidences, argmax path for the string.
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    path = probs.argmax(axis=1)
    chars: list[str] = []
    confs: list[float] = []
    prev = BLANK
    for t, c in enumerate(path.tolist()):
        if c != BLANK and c != prev:
            chars.append(CHARS[c - 1])
            confs.append(float(probs[t, c]))
        prev = c
    text = "".join(chars)
    conf = float(sum(confs) / len(confs)) if confs else 0.0
    return text, conf


class LPRNetMLX:
    """LPRNet backbone + classification head built from MLX primitives.

    Input:  (1, 24, 94, 1) NHWC grayscale plate (MLX default layout).
    Output: (T, NUM_CLASSES) log-probs over T ~= W/8 time steps.
    """

    def __init__(self, dropout_p: float = 0.0) -> None:
        try:
            import mlx.nn as nn
        except Exception as exc:
            raise ImportError(
                "MLX is required for the LPRNet backend. On Apple Silicon: "
                "pip install mlx (see requirements-m5.txt)."
            ) from exc

        def conv_bn_relu(i: int, o: int, k: int = 3, s: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm(o),
                nn.ReLU(),
            )

        self._nn = nn
        self.features = nn.Sequential(
            conv_bn_relu(1, 64),                       # 94x24
            nn.MaxPool2d(kernel_size=3, stride=2),     # ~47x12
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),     # ~24x6
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # squeeze height
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
        )
        self._model = nn.Sequential(self.features, self.classifier)

    def load_weights(self, path: str | Path) -> None:
        import mlx.core as mx

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"LPRNet MLX weights not found: {p}")
        raw = np.load(str(p))
        params: dict[str, mx.array] = {}
        for k in raw.files:
            params[k] = mx.array(raw[k])
        try:
            self._model.load_weights(list(params.items()))
        except Exception:
            # Flat-dict variant for hand-exported checkpoints.
            self._model.update(params)  # type: ignore[attr-defined]
        logger.info("LPRNet MLX weights loaded: %s (%d tensors)", p, len(params))

    def eval(self) -> None:
        self._model.eval()

    def __call__(self, img_nhwc: np.ndarray) -> np.ndarray:
        """Run the graph. Returns (T, NUM_CLASSES) log-probs as numpy."""
        import mlx.core as mx

        x = mx.array(img_nhwc.astype(np.float32))
        logits = self._model(x)          # (1, H', W', C)
        mx.eval(logits)
        arr = np.array(logits)
        # Canonical LPRNet head: average over the height axis so each
        # remaining column is one CTC time step.
        seq = arr.mean(axis=1)[0]        # (W', C)
        # Log-softmax for CTC-compatible scores.
        seq = seq - seq.max(axis=1, keepdims=True)
        e = np.exp(seq)
        return np.log(e / e.sum(axis=1, keepdims=True) + 1e-12)


def preprocess_for_lprnet(crop_bgr: np.ndarray, width: int = 94, height: int = 24) -> np.ndarray:
    """BGR crop -> (1, 24, 94, 1) float32 NHWC grayscale in [0, 1]."""
    try:
        import cv2

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        rs = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image

        rgb = Image.fromarray(crop_bgr[:, :, ::-1]).convert("L")
        rs = np.asarray(rgb.resize((width, height), Image.BILINEAR))
    return (rs.astype(np.float32)[None, :, :, None] / 255.0)

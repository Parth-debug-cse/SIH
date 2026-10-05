"""Plate preprocessing: homography deskew + super-resolution upscale.

Pipeline per crop (BGR uint8 ndarray in/out):
  1. Deskew — explicit quad (from a corner head) warped by homography to a
     fronto-parallel rectangle; else automatic ``minAreaRect`` rectification
     when OpenCV is present; else pass-through with ``deskew_applied=False``.
  2. Upscale — Real-ESRGAN on MPS when torch + weights exist; else
     Lanczos/Cubic resampling + unsharp masking (OpenCV when present,
     PIL/numpy otherwise). A ~20px-tall crop leaves here at ~100px.

No step ever raises for a degenerate input: worst case is the raw crop
returned with the backend name recorded, so the recognizer still runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    target_height: int = 100      # 20px -> ~100px
    max_scale: float = 6.0        # clamp runaway upscales
    esrgan_weights: str = "models/ocr/realesr-general-x4v3.pt"
    esrgan_scale: int = 4
    unsharp_radius: float = 2.0
    unsharp_strength: int = 60    # percent (PIL semantics)


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def _perspective_coeffs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Inverse perspective coefficients for PIL (numpy-only, no cv2)."""
    m = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = float(dst[i][0]), float(dst[i][1])
        u, v = float(src[i][0]), float(src[i][1])
        m[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        m[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v
    return np.linalg.solve(m, b)


class PlatePreprocessor:
    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()
        self._esrgan = None
        self._esrgan_ready = False
        self.upscale_backend = "unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self, crop_bgr: np.ndarray, quad: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """Deskew then upscale. Returns (processed_bgr, info_dict)."""
        flat = np.asarray(crop_bgr)
        if flat.ndim != 3 or flat.shape[2] != 3 or flat.size == 0:
            raise ValueError("crop_bgr must be a non-empty HxWx3 array")

        deskewed, angle, applied = self.deskew(flat, quad=quad)
        upscaled, backend = self.upscale(deskewed)
        h, w = flat.shape[:2]
        uh, uw = upscaled.shape[:2]
        info = {
            "scale": float(uh / max(h, 1)),
            "output_size": [int(uw), int(uh)],
            "deskew_applied": bool(applied),
            "deskew_angle_deg": None if angle is None else float(angle),
            "upscale_backend": backend,
        }
        return upscaled, info

    # ------------------------------------------------------------------
    # 1. Deskew
    # ------------------------------------------------------------------
    def deskew(
        self, crop_bgr: np.ndarray, quad: np.ndarray | None = None
    ) -> tuple[np.ndarray, float | None, bool]:
        """Returns (warped, angle_deg|None, applied)."""
        if quad is not None:
            try:
                return self._warp_quad(crop_bgr, np.asarray(quad, dtype=np.float64)), None, True
            except Exception:
                logger.exception("Quad warp failed; using raw crop")
                return crop_bgr, None, False
        if _has_cv2():
            try:
                return self._auto_deskew_cv2(crop_bgr)
            except Exception:
                logger.exception("Auto deskew failed; using raw crop")
        return crop_bgr, None, False

    def _warp_quad(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Warp an ordered quad [tl, tr, br, bl] to a rectangle.

        Canonical size keeps the quad's own edge lengths (no distortion),
        so a 20px-tall slanted plate becomes a 20px-tall straight plate
        ready for the upscaler.
        """
        if quad.shape != (4, 2):
            raise ValueError("quad must be shape (4, 2), ordered tl/tr/br/bl")
        top = float(np.linalg.norm(quad[1] - quad[0]))
        bottom = float(np.linalg.norm(quad[2] - quad[3]))
        left = float(np.linalg.norm(quad[3] - quad[0]))
        right = float(np.linalg.norm(quad[2] - quad[1]))
        W, H = max(int(round(max(top, bottom))), 8), max(int(round(max(left, right))), 8)
        dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
        if _has_cv2():
            import cv2

            M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32))
            return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
        # numpy + PIL path: PIL needs forward coeffs dst->src.
        from PIL import Image

        coeffs = _perspective_coeffs(dst, quad)
        rgb = Image.fromarray(img[:, :, ::-1])
        warped = rgb.transform((W, H), Image.PERSPECTIVE, coeffs.tolist(), Image.BICUBIC)
        return np.asarray(warped)[:, :, ::-1]

    def _auto_deskew_cv2(self, img: np.ndarray) -> tuple[np.ndarray, float | None, bool]:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img, None, False
        cnt = max(cnts, key=cv2.contourArea)
        if float(cv2.contourArea(cnt)) < 0.05 * img.shape[0] * img.shape[1]:
            return img, None, False  # no dominant plate blob; don't guess
        (cx, cy), (bw, bh), angle = cv2.minAreaRect(cnt)
        if abs(angle) < 1.0 or abs(abs(angle) - 90.0) < 1.0:
            return img, float(angle), False  # already straight
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        h, w = img.shape[:2]
        straight = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return straight, float(angle), True

    # ------------------------------------------------------------------
    # 2. Upscale
    # ------------------------------------------------------------------
    def upscale(self, img_bgr: np.ndarray) -> tuple[np.ndarray, str]:
        h = int(img_bgr.shape[0])
        scale = min(self.config.target_height / max(h, 1),
                    self.config.max_scale)
        if scale <= 1.0:
            self.upscale_backend = "passthrough"
            return img_bgr, self.upscale_backend

        esr = self._try_esrgan(img_bgr, scale)
        if esr is not None:
            self.upscale_backend = "realesrgan_mps"
            return esr, self.upscale_backend

        out = self._resample_upscale(img_bgr, scale)
        self.upscale_backend = "lanczos_unsharp_cv2" if _has_cv2() else "lanczos_unsharp_pil"
        return out, self.upscale_backend

    def _try_esrgan(self, img: np.ndarray, scale: float) -> np.ndarray | None:
        """Real-ESRGAN x4 on MPS. Returns None unless torch + package +
        weights are ALL present — never raises into the caller."""
        if self._esrgan_ready:
            pass  # fall through and reuse self._esrgan (None means unavailable)
        elif not Path(self.config.esrgan_weights).exists():
            self._esrgan_ready = True
            self._esrgan = None
            return None
        else:
            try:
                import torch
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet

                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    logger.warning("ESRGAN weights exist but MPS is unavailable; skipping")
                    self._esrgan_ready, self._esrgan = True, None
                    return None
                model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=self.config.esrgan_scale)
                self._esrgan = RealESRGANer(
                    scale=self.config.esrgan_scale, model_path=self.config.esrgan_weights,
                    model=model, device="mps", pre_pad=0, half=True,
                )
                self._esrgan_ready = True
            except Exception:
                logger.warning("Real-ESRGAN unavailable; using resampling path", exc_info=True)
                self._esrgan_ready, self._esrgan = True, None
                return None
        if self._esrgan is None:
            return None
        try:
            import cv2

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out, _ = self._esrgan.enhance(rgb, outscale=min(scale, self.config.esrgan_scale))
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception:
            logger.exception("ESRGAN enhance failed; using resampling path")
            return None

    def _resample_upscale(self, img: np.ndarray, scale: float) -> np.ndarray:
        h, w = img.shape[:2]
        W, H = int(round(w * scale)), int(round(h * scale))
        if _has_cv2():
            import cv2

            up = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(up, (0, 0), self.config.unsharp_radius)
            sharp = cv2.addWeighted(up, 1.6, blur, -0.6, 0)
            return sharp
        from PIL import Image, ImageFilter

        rgb = Image.fromarray(img[:, :, ::-1])
        up = rgb.resize((W, H), Image.LANCZOS)
        up = up.filter(ImageFilter.UnsharpMask(
            radius=self.config.unsharp_radius,
            percent=self.config.unsharp_strength, threshold=2))
        return np.asarray(up)[:, :, ::-1]

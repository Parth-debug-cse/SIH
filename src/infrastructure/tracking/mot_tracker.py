"""Multi-Object Tracking: OC-SORT-style motion + Re-ID appearance fusion.

Association cascade per frame:
  Stage 1 — motion: IoU match of detections to predicted boxes (Hungarian
      when scipy is present, greedy otherwise).
  Stage 2 — appearance (THE OCR-outage path): unmatched detections are
      matched to unmatched/lost tracks by cosine similarity of the
      vehicle Re-ID embedding. If OCR fails in frame N, the plate text is
      gone but the visual vector is not — the same ``track_id`` survives.

Each confirmed track owns an EMA-smoothed embedding (``embedding`` field):
the binding of visual identity to Track ID. New detections adopt the
matched track's ID only when cosine sim clears ``reid_thresh``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from src.infrastructure.reid.vehicle_reid import VehicleReID, cosine_similarity_matrix

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two [N,4]/[M,4] LTRB box sets."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0, None], boxes_a[:, 1, None], boxes_a[:, 2, None], boxes_a[:, 3, None]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = np.maximum(ax2 - ax1, 0) * np.maximum(ay2 - ay1, 0)
    area_b = np.maximum(bx2 - bx1, 0) * np.maximum(by2 - by1, 0)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Min-cost matching. Hungarian via scipy when present, greedy fallback."""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        pairs: list[tuple[int, int]] = []  # greedy fallback (no scipy needed)
        used_r, used_c = set(), set()
        flat = sorted(
            ((float(cost[r, c]), r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1]))
        )
        for _, r, c in flat:
            if r not in used_r and c not in used_c:
                pairs.append((r, c))
                used_r.add(r)
                used_c.add(c)
        return pairs


# ----------------------------------------------------------------------
# SORT-style Kalman filter (constant velocity, numpy)
# ----------------------------------------------------------------------
class _KalmanBox:
    """7-state [cx, cy, s, r, vx, vy, vs] filter over LTRB boxes."""

    def __init__(self, bbox: list[float]) -> None:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        self.x = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [w * h], [w / max(h, 1e-6)], [0], [0], [0]],
                          dtype=np.float32)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(np.float32)
        self.F = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.diag([1, 1, 10, 10]).astype(np.float32)
        self.Q = np.eye(7, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

    def predict(self) -> list[float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox()

    def update(self, bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        z = np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [w * h], [w / max(h, 1e-6)]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7, dtype=np.float32) - K @ self.H) @ self.P
        return self._to_bbox()

    def _to_bbox(self) -> list[float]:
        cx, cy, s, r = (float(v) for v in self.x[:4, 0])
        w, h = (s * max(r, 1e-6)) ** 0.5, (s / max(r, 1e-6)) ** 0.5
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ----------------------------------------------------------------------
# Track + config
# ----------------------------------------------------------------------
@dataclass
class ReIDTrackerConfig:
    max_age: int = 30          # frames a lost track is kept for Re-ID revival
    min_hits: int = 3          # detections before a track is confirmed
    iou_thresh: float = 0.30   # stage-1 motion gate
    reid_thresh: float = 0.45  # stage-2 cosine-sim gate (appearance)
    embedding_momentum: float = 0.9  # EMA weight for stored track embedding
    det_conf_thresh: float = 0.0     # drop detections below this confidence


@dataclass
class _Track:
    track_id: int
    kf: _KalmanBox
    bbox: list[float]
    embedding: np.ndarray | None = None  # L2-normalized, bound to track_id
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    class_name: str = ""
    confidence: float = 0.0

    def predict(self) -> None:
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def refresh(self, bbox: list[float], emb: np.ndarray | None,
                momentum: float, conf: float, cls: str) -> None:
        self.bbox = self.kf.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.confidence = conf
        self.class_name = cls or self.class_name
        if emb is not None:
            n = float(np.linalg.norm(emb))
            emb = emb / (n if n > 1e-9 else 1.0)
            if self.embedding is None:
                self.embedding = emb.astype(np.float32)
            else:
                fused = momentum * self.embedding + (1.0 - momentum) * emb
                n2 = float(np.linalg.norm(fused))
                self.embedding = (fused / (n2 if n2 > 1e-9 else 1.0)).astype(np.float32)


# ----------------------------------------------------------------------
class ReIDTracker:
    """OC-SORT-style MOT with Re-ID embedding bound to every Track ID.

    Usage per frame::

        tracks = tracker.update(detections, frame=frame)
        # detections: [{bbox, confidence, class_name, embedding? (optional)}]
        # tracks:     [{track_id, bbox, confidence, class_name, embedding}]

    Pass ``frame`` (BGR ndarray) and embeddings are extracted internally and
    bound automatically. Pass precomputed ``embedding`` per detection to skip
    that. ``ocr_ok=False`` entries are still tracked — identity then rides
    purely on cosine similarity (Stage 2).
    """

    def __init__(self, config: ReIDTrackerConfig | None = None,
                 reid: VehicleReID | None = None) -> None:
        self.config = config or ReIDTrackerConfig()
        self.reid = reid or VehicleReID()
        self._tracks: list[_Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    @property
    def active_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks if t.confirmed and t.time_since_update == 0]

    def update(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]:
        cfg = self.config
        dets = [d for d in detections if float(d.get("confidence", 0.0)) >= cfg.det_conf_thresh]

        # 1. Aligned per-detection embeddings (None where unavailable).
        #    Missing vectors simply skip Stage 2 — motion-only for those.
        emb_list = self._embeddings_for(dets, frame)

        # 2. Predict all existing tracks (OC-SORT observation-centric: dead
        #    tracks keep their last observation for re-match).
        for t in self._tracks:
            t.predict()

        # 3. Stage 1 — motion (IoU).
        track_boxes = np.array([t.bbox for t in self._tracks], dtype=np.float32) if self._tracks else np.zeros((0, 4), np.float32)
        det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32) if dets else np.zeros((0, 4), np.float32)
        iou = iou_matrix(track_boxes, det_boxes)
        cost = np.where(iou >= cfg.iou_thresh, 1.0 - iou, 1e6)
        matched, used_t, used_d = [], set(), set()
        for r, c in _linear_assignment(cost):
            if cost[r, c] > 1e5:
                continue
            matched.append((r, c))
            used_t.add(r)
            used_d.add(c)

        # 4. Stage 2 — appearance: unmatched dets vs unmatched tracks by
        #    cosine similarity (OCR-failure / occlusion path).
        un_t = [i for i in range(len(self._tracks)) if i not in used_t]
        un_d = [i for i in range(len(dets)) if i not in used_d]
        if un_t and un_d:
            bank, bank_idx = [], []
            for ti in un_t:
                if self._tracks[ti].embedding is not None:
                    bank.append(self._tracks[ti].embedding)
                    bank_idx.append(ti)
            # Only unmatched detections that actually carry an embedding
            # can be appearance-matched; the rest stay motion-only.
            q_embs, q_cols = [], []
            for c, di in enumerate(un_d):
                e = emb_list[di]
                if e is not None:
                    q_embs.append(e)
                    q_cols.append(c)
            if bank and q_embs:
                sims = cosine_similarity_matrix(np.stack(bank), np.stack(q_embs))
                cost2 = np.where(sims >= cfg.reid_thresh, 1.0 - sims, 1e6)
                for r, c in _linear_assignment(cost2):
                    if cost2[r, c] > 1e5:
                        continue
                    ti, di = bank_idx[r], un_d[q_cols[c]]
                    matched.append((ti, di))
                    used_t.add(ti)
                    used_d.add(di)

        # 5. Refresh matched, birth new, age the missed.
        for ti, di in matched:
            t = self._tracks[ti]
            t.refresh(list(dets[di]["bbox"]), emb_list[di], cfg.embedding_momentum,
                      float(dets[di].get("confidence", 0.0)), str(dets[di].get("class_name", "")))
            if t.hits >= cfg.min_hits:
                t.confirmed = True
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            emb_arr = emb_list[di]
            if emb_arr is not None:
                n = float(np.linalg.norm(emb_arr))
                emb_arr = (emb_arr / (n if n > 1e-9 else 1.0)).astype(np.float32)
            kf = _KalmanBox(list(d["bbox"]))
            t = _Track(track_id=self._next_id, kf=kf, bbox=list(d["bbox"]),
                       embedding=emb_arr, confidence=float(d.get("confidence", 0.0)),
                       class_name=str(d.get("class_name", "")))
            self._next_id += 1
            if cfg.min_hits <= 1:
                t.confirmed = True
            self._tracks.append(t)

        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= cfg.max_age
        ]
        for t in self._tracks:
            if t.time_since_update > 0:
                logger.debug("Track %d missed %d frame(s)",
                             t.track_id, t.time_since_update)

        return [
            {"track_id": t.track_id, "bbox": [float(v) for v in t.bbox],
             "confidence": t.confidence, "class_name": t.class_name,
             "embedding": t.embedding,
             "time_since_update": t.time_since_update}
            for t in self._tracks if t.confirmed and t.time_since_update == 0
        ]

    async def aupdate(self, detections: list[dict], frame: np.ndarray | None = None) -> list[dict]:
        """Async wrapper — embedding extraction runs off the event loop."""
        if frame is not None and any("embedding" not in d for d in detections):
            crops = [_crop(frame, d["bbox"]) for d in detections if "embedding" not in d]
            if crops:
                new_embs = await asyncio.to_thread(self.reid.extract, crops)
                j = 0
                for d in detections:
                    if "embedding" not in d:
                        d["embedding"] = new_embs[j]
                        j += 1
            return self.update(detections, frame=None)
        return self.update(detections, frame=frame)

    # ------------------------------------------------------------------
    def _embeddings_for(
        self, dets: list[dict], frame: np.ndarray | None
    ) -> list[np.ndarray | None]:
        """Aligned per-detection embeddings; None where unavailable.

        Precomputed ``embedding`` entries are used as-is; missing ones are
        extracted from ``frame`` crops via the Re-ID backbone. Detections
        with no vector stay motion-only (they skip Stage 2, never crash it).
        """
        out: list[np.ndarray | None] = [
            np.asarray(d.get("embedding"), dtype=np.float32)
            if d.get("embedding") is not None else None
            for d in dets
        ]
        need = [i for i, e in enumerate(out) if e is None]
        if need and frame is not None:
            crops = [_crop(frame, dets[i]["bbox"]) for i in need]
            fresh = self.reid.extract(crops)
            for k, i in enumerate(need):
                out[i] = fresh[k].astype(np.float32)
                dets[i]["embedding"] = out[i]  # bind for downstream (OCR vote, etc.)
        return out


def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]

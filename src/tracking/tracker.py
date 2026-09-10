import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_age=30, n_init=3):
        self.max_age = max_age
        self.n_init = n_init
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections, frame=None, frame_shape=None):
        """Update tracker with new detections.

        Args:
            detections: list of dicts with bbox, confidence, class_name
            frame: actual numpy frame (preferred for appearance features)
            frame_shape: (height, width) tuple as fallback
        """
        if not detections:
            return []

        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            raw_dets.append(
                ([x1, y1, x2 - x1, y2 - y1], det["confidence"], det.get("class_name", ""))
            )

        if frame is not None:
            tracks = self.tracker.update_tracks(raw_dets, frame=frame)
        elif frame_shape is not None:
            blank = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
            tracks = self.tracker.update_tracks(raw_dets, frame=blank)
        else:
            tracks = self.tracker.update_tracks(raw_dets)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            results.append({
                "track_id": int(track.track_id),
                "bbox": [float(l), float(t), float(r), float(b)],
                "confidence": float(track.get_det_conf()) if track.get_det_conf() else 0.0,
                "class_name": str(track.get_det_class()) if track.get_det_class() else "",
            })

        return results

    def reset(self):
        self.tracker = DeepSort(max_age=self.max_age, n_init=self.n_init)

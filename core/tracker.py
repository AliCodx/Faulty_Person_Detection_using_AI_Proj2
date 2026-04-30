# core/tracker.py

import numpy as np

class Tracker:
    """
    Simple IOU-based tracker (lightweight production baseline).
    Can later be replaced with DeepSORT or ByteTrack.
    """

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}   # id -> bbox
        self.next_id = 0

    # -----------------------------
    # IOU CALCULATION
    # -----------------------------
    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1p, y1p, x2p, y2p = box2

        xi1 = max(x1, x1p)
        yi1 = max(y1, y1p)
        xi2 = min(x2, x2p)
        yi2 = min(y2, y2p)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2p - x1p) * (y2p - y1p)

        union = box1_area + box2_area - inter_area

        if union == 0:
            return 0

        return inter_area / union

    # -----------------------------
    # MAIN UPDATE FUNCTION
    # -----------------------------
    def update(self, detections):
        """
        detections: list of bounding boxes
        returns: list of (track_id, box)
        """

        assigned_tracks = []
        used_tracks = set()

        for box in detections:
            best_id = None
            best_iou = 0

            # match with existing tracks
            for tid, tbox in self.tracks.items():
                if tid in used_tracks:
                    continue

                score = self.iou(box, tbox)

                if score > best_iou and score > self.iou_threshold:
                    best_iou = score
                    best_id = tid

            # assign ID
            if best_id is None:
                best_id = f"T{self.next_id}"
                self.next_id += 1

            self.tracks[best_id] = box
            used_tracks.add(best_id)

            assigned_tracks.append((best_id, box))

        return assigned_tracks
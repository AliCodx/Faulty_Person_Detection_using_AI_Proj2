class Tracker:
    """
    Simple IOU-based tracker with stale-track cleanup.
    Keeps detections stable within the same camera view.
    """

    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}   # id -> {"box": bbox, "missed": int}
        self.next_id = 0

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

    def update(self, detections):
        detections = [tuple(box) for box in detections]
        assigned_tracks = {}
        matched_tracks = set()
        matched_detections = set()

        candidates = []
        for det_idx, box in enumerate(detections):
            for tid, track in self.tracks.items():
                score = self.iou(box, track["box"])
                if score >= self.iou_threshold:
                    candidates.append((score, det_idx, tid))

        candidates.sort(reverse=True)

        for score, det_idx, tid in candidates:
            if det_idx in matched_detections or tid in matched_tracks:
                continue

            matched_detections.add(det_idx)
            matched_tracks.add(tid)
            assigned_tracks[det_idx] = tid
            self.tracks[tid]["box"] = detections[det_idx]
            self.tracks[tid]["missed"] = 0

        for tid, track in list(self.tracks.items()):
            if tid not in matched_tracks:
                track["missed"] += 1

        results = []
        for det_idx, box in enumerate(detections):
            if det_idx not in assigned_tracks:
                tid = f"T{self.next_id}"
                self.next_id += 1
                self.tracks[tid] = {"box": box, "missed": 0}
                assigned_tracks[det_idx] = tid
            results.append((assigned_tracks[det_idx], box))

        stale_ids = [
            tid
            for tid, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        return results

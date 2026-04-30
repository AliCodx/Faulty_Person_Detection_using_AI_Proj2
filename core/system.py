import cv2
import time


class TrackingSystem:
    def __init__(self, detector, extractor, id_manager, registry, db, event_cooldown=2.0):
        self.detector = detector
        self.extractor = extractor
        self.id_manager = id_manager
        self.registry = registry
        self.db = db
        self.event_cooldown = event_cooldown
        self.last_seen_log = {}

    def _crop_person(self, frame, box):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def _encode_snapshot(self, crop):
        if crop is None or crop.size == 0:
            return None

        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            return None
        return buffer.tobytes()

    def _should_log_seen(self, pid, cam_id):
        now = time.time()
        key = (pid, cam_id)
        last_time = self.last_seen_log.get(key, 0)
        if now - last_time >= self.event_cooldown:
            self.last_seen_log[key] = now
            return True
        return False

    def process(self, frame, cam_id):
        boxes = self.detector.detect(frame)
        results = []

        for box in boxes:
            crop = self._crop_person(frame, box)
            embedding = self.extractor.extract(crop)
            pid = self.id_manager.get_id(embedding)
            snapshot = self._encode_snapshot(crop)

            if pid not in self.registry.memory:
                self.registry.add(pid, embedding, snapshot=snapshot, camera_id=cam_id)
            else:
                self.registry.update_seen(
                    pid,
                    embedding=embedding,
                    snapshot=snapshot,
                    camera_id=cam_id,
                )

            faulty = self.registry.is_faulty(pid)
            if self._should_log_seen(pid, cam_id):
                event_name = "FAULTY_SEEN" if faulty else "SEEN"
                self.db.log_event(pid, cam_id, event_name)

            results.append(
                {
                    "box": [int(v) for v in box],
                    "pid": pid,
                    "faulty": faulty,
                }
            )

        return results

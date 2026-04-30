import cv2
import time

from core.tracker import Tracker


class TrackingSystem:
    def __init__(
        self,
        detector,
        extractor,
        id_manager,
        registry,
        db,
        event_cooldown=2.0,
        tracker_iou_threshold=0.35,
        tracker_max_missed=20,
    ):
        self.detector = detector
        self.extractor = extractor
        self.id_manager = id_manager
        self.registry = registry
        self.db = db
        self.event_cooldown = event_cooldown
        self.tracker_iou_threshold = tracker_iou_threshold
        self.tracker_max_missed = tracker_max_missed
        self.last_seen_log = {}
        self.trackers = {}
        self.track_to_pid = {}

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

    def _get_tracker(self, cam_id):
        if cam_id not in self.trackers:
            self.trackers[cam_id] = Tracker(
                iou_threshold=self.tracker_iou_threshold,
                max_missed=self.tracker_max_missed,
            )
        if cam_id not in self.track_to_pid:
            self.track_to_pid[cam_id] = {}
        return self.trackers[cam_id]

    def _cleanup_stale_track_mappings(self, cam_id):
        active_track_ids = set(self.trackers[cam_id].tracks.keys())
        current_map = self.track_to_pid.setdefault(cam_id, {})
        stale_track_ids = [
            track_id
            for track_id in current_map
            if track_id not in active_track_ids
        ]
        for track_id in stale_track_ids:
            del current_map[track_id]

    def _remap_pid_everywhere(self, old_pid, new_pid):
        if old_pid == new_pid:
            return

        for cam_id, track_map in self.track_to_pid.items():
            for track_id, mapped_pid in list(track_map.items()):
                if mapped_pid == old_pid:
                    track_map[track_id] = new_pid

    def _merge_duplicate_identity(self, pid, embedding):
        best_other_pid, best_other_score = self.id_manager.find_best_match(
            embedding,
            exclude_pid=pid,
        )
        if best_other_pid is None:
            return pid

        if not self.id_manager.should_merge(pid, best_other_pid, best_other_score):
            return pid

        canonical_pid = self.registry.choose_canonical_pid(pid, best_other_pid)
        duplicate_pid = best_other_pid if canonical_pid == pid else pid
        canonical_pid = self.registry.merge_people(duplicate_pid, canonical_pid)
        self._remap_pid_everywhere(duplicate_pid, canonical_pid)
        return canonical_pid

    def process(self, frame, cam_id):
        boxes = self.detector.detect(frame)
        tracker = self._get_tracker(cam_id)
        tracked_boxes = tracker.update(boxes)
        self._cleanup_stale_track_mappings(cam_id)
        results = []

        for track_id, box in tracked_boxes:
            crop = self._crop_person(frame, box)
            embedding = self.extractor.extract(crop)
            current_pid = self.track_to_pid[cam_id].get(track_id)
            pid = self.id_manager.get_id(
                embedding,
                current_pid=current_pid,
                cam_id=cam_id,
            )
            self.track_to_pid[cam_id][track_id] = pid
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

            canonical_pid = self._merge_duplicate_identity(pid, embedding)
            if canonical_pid != pid:
                pid = canonical_pid
                self.track_to_pid[cam_id][track_id] = pid

            faulty = self.registry.is_faulty(pid)
            if self._should_log_seen(pid, cam_id):
                event_name = "FAULTY_SEEN" if faulty else "SEEN"
                self.db.log_event(pid, cam_id, event_name)

            results.append(
                {
                    "box": [int(v) for v in box],
                    "pid": pid,
                    "faulty": faulty,
                    "track_id": track_id,
                }
            )

        return results

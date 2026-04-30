import re

import numpy as np


class IDManager:
    def __init__(
        self,
        registry,
        threshold=0.70,
        track_stickiness_threshold=0.58,
        faulty_match_boost=0.05,
        recent_cross_camera_seconds=4,
        recent_match_boost=0.12,
        id_switch_margin=0.06,
        duplicate_merge_threshold=0.78,
    ):
        self.registry = registry
        self.threshold = threshold
        self.track_stickiness_threshold = track_stickiness_threshold
        self.faulty_match_boost = faulty_match_boost
        self.recent_cross_camera_seconds = recent_cross_camera_seconds
        self.recent_match_boost = recent_match_boost
        self.id_switch_margin = id_switch_margin
        self.duplicate_merge_threshold = duplicate_merge_threshold
        self.next_id = self._get_next_id()

    def _get_next_id(self):
        highest = -1
        for pid in self.registry.memory:
            match = re.fullmatch(r"P(\d+)", pid)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def cosine(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if a.ndim != 1 or b.ndim != 1:
            a = a.flatten()
            b = b.flatten()

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a, b) / (a_norm * b_norm))

    def score_against_person(self, embedding, person_data):
        scores = []

        prototype = person_data.get("embedding")
        if prototype is not None:
            scores.append(self.cosine(embedding, prototype))

        for gallery_embedding in person_data.get("gallery", []):
            if gallery_embedding is not None:
                scores.append(self.cosine(embedding, gallery_embedding))

        return max(scores) if scores else 0.0

    def score_for_pid(self, embedding, pid):
        person_data = self.registry.memory.get(pid)
        if person_data is None:
            return 0.0
        return self.score_against_person(embedding, person_data)

    def _match_threshold_for_pid(self, pid, cam_id=None):
        threshold = self.threshold
        if self.registry.is_faulty(pid):
            threshold = max(0.0, threshold - self.faulty_match_boost)
        if cam_id is not None and self.registry.was_seen_recently_elsewhere(
            pid,
            cam_id,
            seconds=self.recent_cross_camera_seconds,
        ):
            threshold = max(0.0, threshold - self.recent_match_boost)
        return threshold

    def _track_threshold_for_pid(self, pid, cam_id=None):
        threshold = self.track_stickiness_threshold
        if self.registry.is_faulty(pid):
            threshold = max(0.0, threshold - self.faulty_match_boost)
        if cam_id is not None and self.registry.was_seen_recently_elsewhere(
            pid,
            cam_id,
            seconds=self.recent_cross_camera_seconds,
        ):
            threshold = max(0.0, threshold - self.recent_match_boost)
        return threshold

    def find_best_match(self, embedding, exclude_pid=None):
        best_id = None
        best_score = -1.0

        for pid, data in self.registry.memory.items():
            if exclude_pid is not None and pid == exclude_pid:
                continue
            sim = self.score_against_person(embedding, data)
            if sim > best_score:
                best_score = sim
                best_id = pid

        return best_id, best_score

    def should_merge(self, pid_a, pid_b, score):
        if pid_a == pid_b:
            return False

        threshold = self.duplicate_merge_threshold
        if self.registry.is_faulty(pid_a) or self.registry.is_faulty(pid_b):
            threshold -= self.faulty_match_boost

        return score >= max(0.0, threshold)

    def get_id(self, embedding, current_pid=None, cam_id=None):
        current_score = -1.0
        if current_pid is not None:
            current_score = self.score_for_pid(embedding, current_pid)
        else:
            current_score = -1.0

        best_id, best_score = self.find_best_match(embedding)

        if current_pid is not None and best_id is not None and best_id != current_pid:
            if best_score >= self._match_threshold_for_pid(best_id, cam_id) and (
                current_score < self._track_threshold_for_pid(current_pid, cam_id)
                or best_score >= current_score + self.id_switch_margin
            ):
                return best_id

        if current_pid is not None and current_score >= self._track_threshold_for_pid(current_pid, cam_id):
            return current_pid

        if best_id is not None and best_score >= self._match_threshold_for_pid(best_id, cam_id):
            return best_id

        pid = f"P{self.next_id}"
        self.next_id += 1
        return pid

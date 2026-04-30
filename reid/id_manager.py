import re

import numpy as np


class IDManager:
    def __init__(self, registry, threshold=0.82):
        self.registry = registry
        self.threshold = threshold
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

    def get_id(self, embedding):
        best_id = None
        best_score = -1.0

        for pid, data in self.registry.memory.items():
            known_embedding = data.get("embedding")
            if known_embedding is None:
                continue

            sim = self.cosine(embedding, known_embedding)
            if sim > best_score and sim >= self.threshold:
                best_score = sim
                best_id = pid

        if best_id is not None:
            return best_id

        pid = f"P{self.next_id}"
        self.next_id += 1
        return pid

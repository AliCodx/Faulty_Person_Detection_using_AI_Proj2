import numpy as np
from datetime import datetime, timedelta


class Registry:
    def __init__(self, db, gallery_size=6, blend_alpha=0.65):
        self.gallery_size = gallery_size
        self.blend_alpha = blend_alpha
        self.db = db
        self.memory = {}
        self.load_from_db()

    def _normalize_embedding(self, embedding):
        if embedding is None:
            return None

        vector = np.asarray(embedding, dtype=np.float32).flatten()
        if vector.size == 0:
            return None

        norm = np.linalg.norm(vector)
        if norm == 0:
            return None

        return vector / norm

    def _cosine(self, a, b):
        if a is None or b is None:
            return 0.0

        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        if a.shape != b.shape or a.size == 0:
            return 0.0

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a, b) / (a_norm * b_norm))
    def _build_gallery(self, embedding):
        if embedding is None:
            return []
        return [embedding.copy()]

    def _append_gallery(self, gallery, embedding):
        if embedding is None:
            return list(gallery or [])

        updated_gallery = [item for item in (gallery or []) if item is not None]
        if not updated_gallery or self._cosine(updated_gallery[-1], embedding) < 0.995:
            updated_gallery.append(embedding.copy())

        return updated_gallery[-self.gallery_size :]

    def _blend_embeddings(self, old_embedding, new_embedding):
        if new_embedding is None:
            return old_embedding

        if old_embedding is None:
            return new_embedding

        old_embedding = self._normalize_embedding(old_embedding)
        new_embedding = self._normalize_embedding(new_embedding)
        if old_embedding is None:
            return new_embedding
        if new_embedding is None:
            return old_embedding
        if old_embedding.shape != new_embedding.shape:
            return new_embedding

        blended = (self.blend_alpha * old_embedding) + ((1.0 - self.blend_alpha) * new_embedding)
        return self._normalize_embedding(blended)

    def _parse_timestamp(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def load_from_db(self):
        self.memory = {}
        for person in self.db.get_all_persons():
            self.memory[person["id"]] = self._to_memory_entry(person)

    def _to_memory_entry(self, person):
        embedding = self._normalize_embedding(person["embedding"])
        return {
            "embedding": embedding,
            "gallery": self._build_gallery(embedding),
            "faulty": bool(person["is_faulty"]),
            "note": person["note"] or "",
            "first_seen": person["first_seen"],
            "last_seen": person["last_seen"],
            "last_camera": person["last_camera"],
            "sightings": int(person["sightings"] or 0),
            "snapshot": person["snapshot"],
        }

    def add(self, pid, embedding, snapshot=None, camera_id=None):
        normalized_embedding = self._normalize_embedding(embedding)
        self.db.upsert_person(
            pid,
            embedding=normalized_embedding,
            snapshot=snapshot,
            camera_id=camera_id,
            increment_sighting=True,
        )
        entry = self._to_memory_entry(self.db.get_person(pid))
        entry["gallery"] = self._build_gallery(normalized_embedding)
        self.memory[pid] = entry

    def update_seen(self, pid, embedding=None, snapshot=None, camera_id=None):
        if pid not in self.memory:
            self.add(pid, embedding, snapshot=snapshot, camera_id=camera_id)
            return

        normalized_embedding = self._normalize_embedding(embedding)
        current_entry = self.memory[pid]
        blended_embedding = self._blend_embeddings(
            current_entry.get("embedding"),
            normalized_embedding,
        )
        gallery = self._append_gallery(current_entry.get("gallery", []), normalized_embedding)

        self.db.upsert_person(
            pid,
            embedding=blended_embedding,
            snapshot=snapshot,
            camera_id=camera_id,
            increment_sighting=True,
        )
        updated_entry = self._to_memory_entry(self.db.get_person(pid))
        updated_entry["embedding"] = blended_embedding
        updated_entry["gallery"] = gallery
        self.memory[pid] = updated_entry

    def mark_faulty(self, pid, faulty=True, note=None):
        if pid not in self.memory:
            person = self.db.get_person(pid)
            if person is None:
                return
            self.memory[pid] = self._to_memory_entry(person)

        existing_gallery = list(self.memory[pid].get("gallery", []))
        self.db.mark_faulty(pid, faulty=faulty, note=note)
        event_name = "FAULTY_MARKED" if faulty else "FAULTY_CLEARED"
        self.db.log_event(pid, "ALL", event_name, details=note or "")
        updated_entry = self._to_memory_entry(self.db.get_person(pid))
        updated_entry["gallery"] = existing_gallery or updated_entry["gallery"]
        self.memory[pid] = updated_entry

    def is_faulty(self, pid):
        if pid in self.memory:
            return self.memory[pid]["faulty"]

        person = self.db.get_person(pid)
        if person is None:
            return False

        self.memory[pid] = self._to_memory_entry(person)
        return self.memory[pid]["faulty"]

    def was_seen_recently_elsewhere(self, pid, cam_id, seconds=4):
        person = self.get_person(pid)
        if person is None:
            return False

        last_seen = self._parse_timestamp(person.get("last_seen"))
        if last_seen is None:
            return False

        if person.get("last_camera") is None:
            return False

        if str(person.get("last_camera")) == str(cam_id):
            return False

        return last_seen >= (datetime.now() - timedelta(seconds=seconds))

    def choose_canonical_pid(self, pid_a, pid_b):
        person_a = self.get_person(pid_a)
        person_b = self.get_person(pid_b)
        if person_a is None:
            return pid_b
        if person_b is None:
            return pid_a

        if person_a["faulty"] and not person_b["faulty"]:
            return pid_a
        if person_b["faulty"] and not person_a["faulty"]:
            return pid_b

        first_a = self._parse_timestamp(person_a.get("first_seen"))
        first_b = self._parse_timestamp(person_b.get("first_seen"))
        if first_a and first_b and first_a != first_b:
            return pid_a if first_a <= first_b else pid_b

        def pid_number(pid):
            try:
                return int(str(pid).lstrip("P"))
            except ValueError:
                return 10**9

        return pid_a if pid_number(pid_a) <= pid_number(pid_b) else pid_b

    def merge_people(self, source_pid, target_pid):
        if source_pid == target_pid:
            return target_pid

        source = self.get_person(source_pid)
        target = self.get_person(target_pid)
        if source is None or target is None:
            return target_pid

        source_gallery = list(source.get("gallery", []))
        target_gallery = list(target.get("gallery", []))
        combined_gallery = []
        for embedding in target_gallery + source_gallery:
            combined_gallery = self._append_gallery(combined_gallery, embedding)

        self.db.merge_persons(source_pid, target_pid)
        updated_entry = self._to_memory_entry(self.db.get_person(target_pid))
        updated_entry["gallery"] = combined_gallery or updated_entry["gallery"]
        self.memory[target_pid] = updated_entry
        self.memory.pop(source_pid, None)
        return target_pid

    def get_person(self, pid):
        if pid not in self.memory:
            person = self.db.get_person(pid)
            if person is None:
                return None
            self.memory[pid] = self._to_memory_entry(person)

        return {"id": pid, **self.memory[pid]}

    def all_people(self):
        return [
            {"id": pid, **data}
            for pid, data in sorted(
                self.memory.items(),
                key=lambda item: (item[1]["last_seen"] or "", item[0]),
                reverse=True,
            )
        ]

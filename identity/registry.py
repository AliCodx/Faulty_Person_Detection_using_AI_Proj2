class Registry:
    def __init__(self, db):
        self.db = db
        self.memory = {}
        self.load_from_db()

    def load_from_db(self):
        self.memory = {}
        for person in self.db.get_all_persons():
            self.memory[person["id"]] = self._to_memory_entry(person)

    def _to_memory_entry(self, person):
        return {
            "embedding": person["embedding"],
            "faulty": bool(person["is_faulty"]),
            "note": person["note"] or "",
            "first_seen": person["first_seen"],
            "last_seen": person["last_seen"],
            "last_camera": person["last_camera"],
            "sightings": int(person["sightings"] or 0),
            "snapshot": person["snapshot"],
        }

    def add(self, pid, embedding, snapshot=None, camera_id=None):
        self.db.upsert_person(
            pid,
            embedding=embedding,
            snapshot=snapshot,
            camera_id=camera_id,
            increment_sighting=True,
        )
        self.memory[pid] = self._to_memory_entry(self.db.get_person(pid))

    def update_seen(self, pid, embedding=None, snapshot=None, camera_id=None):
        if pid not in self.memory:
            self.add(pid, embedding, snapshot=snapshot, camera_id=camera_id)
            return

        self.db.upsert_person(
            pid,
            embedding=embedding,
            snapshot=snapshot,
            camera_id=camera_id,
            increment_sighting=True,
        )
        self.memory[pid] = self._to_memory_entry(self.db.get_person(pid))

    def mark_faulty(self, pid, faulty=True, note=None):
        if pid not in self.memory:
            person = self.db.get_person(pid)
            if person is None:
                return
            self.memory[pid] = self._to_memory_entry(person)

        self.db.mark_faulty(pid, faulty=faulty, note=note)
        event_name = "FAULTY_MARKED" if faulty else "FAULTY_CLEARED"
        self.db.log_event(pid, "ALL", event_name, details=note or "")
        self.memory[pid] = self._to_memory_entry(self.db.get_person(pid))

    def is_faulty(self, pid):
        if pid in self.memory:
            return self.memory[pid]["faulty"]

        person = self.db.get_person(pid)
        if person is None:
            return False

        self.memory[pid] = self._to_memory_entry(person)
        return self.memory[pid]["faulty"]

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

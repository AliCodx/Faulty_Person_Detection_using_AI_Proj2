import sqlite3
from datetime import datetime

import numpy as np


class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        self._migrate_tables()

    def create_tables(self):
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                embedding BLOB,
                embedding_dim INTEGER DEFAULT 0,
                is_faulty INTEGER DEFAULT 0,
                note TEXT DEFAULT '',
                first_seen TEXT,
                last_seen TEXT,
                last_camera TEXT,
                sightings INTEGER DEFAULT 0,
                snapshot BLOB
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                camera_id TEXT,
                event_type TEXT,
                timestamp TEXT,
                details TEXT DEFAULT ''
            )
            """
        )

        self.conn.commit()

    def _migrate_tables(self):
        self._ensure_columns(
            "persons",
            {
                "embedding_dim": "INTEGER DEFAULT 0",
                "note": "TEXT DEFAULT ''",
                "first_seen": "TEXT",
                "last_seen": "TEXT",
                "last_camera": "TEXT",
                "sightings": "INTEGER DEFAULT 0",
                "snapshot": "BLOB",
            },
        )
        self._ensure_columns("events", {"details": "TEXT DEFAULT ''"})
        self._backfill_embedding_dims()

    def _backfill_embedding_dims(self):
        self.conn.execute(
            """
            UPDATE persons
            SET embedding_dim = CAST(length(embedding) / 4 AS INTEGER)
            WHERE embedding IS NOT NULL
              AND (embedding_dim IS NULL OR embedding_dim = 0)
              AND (length(embedding) % 4) = 0
            """
        )
        self.conn.commit()

    def _ensure_columns(self, table_name, columns):
        cur = self.conn.cursor()
        existing = {
            row["name"]
            for row in cur.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

        for column_name, definition in columns.items():
            if column_name not in existing:
                cur.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
                )

        self.conn.commit()

    def _to_numpy(self, embedding):
        if embedding is None:
            return None

        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32)

        try:
            import torch

            if isinstance(embedding, torch.Tensor):
                return embedding.detach().cpu().numpy().astype(np.float32)
        except ImportError:
            pass

        return np.array(embedding, dtype=np.float32)

    def _serialize_embedding(self, embedding):
        vector = self._to_numpy(embedding)
        if vector is None:
            return None, 0

        vector = vector.flatten().astype(np.float32)
        return vector.tobytes(), int(vector.size)

    def _deserialize_embedding(self, blob, dim):
        if blob is None:
            return None

        if len(blob) % 4 != 0:
            return None

        vector = np.frombuffer(blob, dtype=np.float32)

        if not dim:
            dim = int(vector.size)

        if dim and vector.size != dim:
            vector = vector[:dim]

        return vector

    def _parse_timestamp(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def upsert_person(
        self,
        pid,
        embedding=None,
        is_faulty=None,
        note=None,
        camera_id=None,
        snapshot=None,
        increment_sighting=False,
    ):
        current = self.get_person(pid)
        now = datetime.now().isoformat(timespec="seconds")

        embedding_blob, embedding_dim = self._serialize_embedding(embedding)

        if current is None:
            faulty_value = int(bool(is_faulty)) if is_faulty is not None else 0
            note_value = note or ""
            sightings = 1 if increment_sighting else 0

            self.conn.execute(
                """
                INSERT INTO persons (
                    id, embedding, embedding_dim, is_faulty, note,
                    first_seen, last_seen, last_camera, sightings, snapshot
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    embedding_blob,
                    embedding_dim,
                    faulty_value,
                    note_value,
                    now,
                    now,
                    str(camera_id) if camera_id is not None else None,
                    sightings,
                    snapshot,
                ),
            )
        else:
            faulty_value = (
                int(bool(is_faulty))
                if is_faulty is not None
                else int(current["is_faulty"])
            )
            note_value = note if note is not None else current["note"]
            updated_blob = embedding_blob if embedding_blob is not None else current["embedding"]
            updated_dim = embedding_dim if embedding_blob is not None else current["embedding_dim"]
            updated_snapshot = snapshot if snapshot is not None else current["snapshot"]
            sightings = int(current["sightings"]) + (1 if increment_sighting else 0)
            last_camera = (
                str(camera_id) if camera_id is not None else current["last_camera"]
            )

            self.conn.execute(
                """
                UPDATE persons
                SET embedding = ?, embedding_dim = ?, is_faulty = ?, note = ?,
                    last_seen = ?, last_camera = ?, sightings = ?, snapshot = ?
                WHERE id = ?
                """,
                (
                    updated_blob,
                    updated_dim,
                    faulty_value,
                    note_value,
                    now,
                    last_camera,
                    sightings,
                    updated_snapshot,
                    pid,
                ),
            )

        self.conn.commit()

    def mark_faulty(self, pid, faulty=True, note=None):
        current = self.get_person(pid)
        if current is None:
            return

        note_value = note if note is not None else current["note"]
        self.conn.execute(
            """
            UPDATE persons
            SET is_faulty = ?, note = ?
            WHERE id = ?
            """,
            (1 if faulty else 0, note_value, pid),
        )
        self.conn.commit()

    def log_event(self, pid, cam, event, details=""):
        self.conn.execute(
            """
            INSERT INTO events (person_id, camera_id, event_type, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pid, str(cam), event, datetime.now().isoformat(timespec="seconds"), details),
        )
        self.conn.commit()

    def get_person(self, pid):
        cur = self.conn.cursor()
        row = cur.execute("SELECT * FROM persons WHERE id = ?", (pid,)).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def get_all_persons(self):
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM persons ORDER BY last_seen DESC, id ASC"
        ).fetchall()
        return [self._row_to_person(row) for row in rows]

    def get_recent_events(self, limit=50):
        cur = self.conn.cursor()
        return cur.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def merge_persons(self, source_pid, target_pid):
        if source_pid == target_pid:
            return

        source = self.get_person(source_pid)
        target = self.get_person(target_pid)
        if source is None or target is None:
            return

        source_first = self._parse_timestamp(source["first_seen"])
        target_first = self._parse_timestamp(target["first_seen"])
        source_last = self._parse_timestamp(source["last_seen"])
        target_last = self._parse_timestamp(target["last_seen"])

        first_seen = source["first_seen"]
        if target_first and (source_first is None or target_first <= source_first):
            first_seen = target["first_seen"]

        last_seen = target["last_seen"]
        last_camera = target["last_camera"]
        if source_last and (target_last is None or source_last > target_last):
            last_seen = source["last_seen"]
            last_camera = source["last_camera"]

        merged_faulty = bool(source["is_faulty"]) or bool(target["is_faulty"])
        merged_note = target["note"] or source["note"] or ""
        merged_sightings = int(source["sightings"] or 0) + int(target["sightings"] or 0)
        merged_snapshot = target["snapshot"] if target["snapshot"] is not None else source["snapshot"]
        merged_embedding = target["embedding"] if target["embedding"] is not None else source["embedding"]
        embedding_blob, embedding_dim = self._serialize_embedding(merged_embedding)

        self.conn.execute(
            """
            UPDATE persons
            SET embedding = ?, embedding_dim = ?, is_faulty = ?, note = ?,
                first_seen = ?, last_seen = ?, last_camera = ?, sightings = ?, snapshot = ?
            WHERE id = ?
            """,
            (
                embedding_blob,
                embedding_dim,
                int(merged_faulty),
                merged_note,
                first_seen,
                last_seen,
                last_camera,
                merged_sightings,
                merged_snapshot,
                target_pid,
            ),
        )
        self.conn.execute(
            "UPDATE events SET person_id = ? WHERE person_id = ?",
            (target_pid, source_pid),
        )
        self.conn.execute("DELETE FROM persons WHERE id = ?", (source_pid,))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def _row_to_person(self, row):
        person = dict(row)
        person["embedding"] = self._deserialize_embedding(
            row["embedding"], row["embedding_dim"]
        )
        person["is_faulty"] = bool(row["is_faulty"])
        return person

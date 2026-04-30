from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_NAME
from db.database import Database
from identity.registry import Registry


st.set_page_config(page_title="Faulty Person Detection Dashboard", layout="wide")


def load_state():
    db = Database(DB_NAME)
    registry = Registry(db)
    return db, registry


def status_badge(person):
    return "Faulty" if person["faulty"] else "Normal"


def show_person_card(person):
    snapshot = person.get("snapshot")
    status = status_badge(person)

    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            if snapshot:
                st.image(snapshot, caption=person["id"], use_container_width=True)
            else:
                st.caption("No snapshot available")

        with right:
            st.subheader(person["id"])
            st.write(f"Status: {status}")
            st.write(f"Sightings: {person.get('sightings', 0)}")
            st.write(f"Last camera: {person.get('last_camera') or 'Unknown'}")
            st.write(f"Last seen: {person.get('last_seen') or 'Unknown'}")
            st.write(f"Note: {person.get('note') or '-'}")


def show_dashboard():
    db, registry = load_state()
    people = registry.all_people()

    st.title("Faulty Person Detection System")
    st.caption(
        "Review detected persons, mark anyone as faulty or suspicious, and monitor recent sightings."
    )

    faulty_count = sum(1 for person in people if person["faulty"])
    total_sightings = sum(person.get("sightings", 0) for person in people)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Known Persons", len(people))
    metric2.metric("Faulty Persons", faulty_count)
    metric3.metric("Total Sightings", total_sightings)

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Detected Persons")
        if not people:
            st.info("No persons stored yet. Start the main camera pipeline first.")
        else:
            options = [person["id"] for person in people]
            selected_pid = st.selectbox("Choose a person", options=options)
            selected_person = next(person for person in people if person["id"] == selected_pid)
            show_person_card(selected_person)

    with right:
        st.subheader("Flag Person")
        if people:
            selected_pid = st.selectbox(
                "Person ID",
                options=[person["id"] for person in people],
                key="action_pid",
            )
            current = next(person for person in people if person["id"] == selected_pid)
            note_value = st.text_area(
                "Reason / Note",
                value=current.get("note", ""),
                placeholder="Explain why this person is suspicious or faulty",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Mark Faulty", use_container_width=True):
                    registry.mark_faulty(selected_pid, faulty=True, note=note_value.strip())
                    st.success(f"{selected_pid} marked as faulty.")
                    st.rerun()
            with col2:
                if st.button("Clear Flag", use_container_width=True):
                    registry.mark_faulty(selected_pid, faulty=False, note=note_value.strip())
                    st.success(f"{selected_pid} cleared.")
                    st.rerun()

    st.divider()
    st.subheader("Recent Events")
    events = db.get_recent_events(limit=50)
    if not events:
        st.info("No events recorded yet.")
    else:
        event_rows = [
            {
                "ID": row["id"],
                "Person": row["person_id"],
                "Camera": row["camera_id"],
                "Event": row["event_type"],
                "Time": row["timestamp"],
                "Details": row["details"],
            }
            for row in events
        ]
        st.dataframe(event_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    show_dashboard()

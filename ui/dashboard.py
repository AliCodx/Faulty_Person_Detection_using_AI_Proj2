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


def init_session_state():
    st.session_state.setdefault("selected_pid", None)
    st.session_state.setdefault("action_pid", None)
    st.session_state.setdefault("action_note", "")
    st.session_state.setdefault("action_note_pid", None)
    st.session_state.setdefault("auto_refresh_enabled", True)


def ensure_valid_selection(people):
    if not people:
        st.session_state["selected_pid"] = None
        st.session_state["action_pid"] = None
        st.session_state["action_note"] = ""
        st.session_state["action_note_pid"] = None
        return

    valid_ids = [person["id"] for person in people]

    if st.session_state["selected_pid"] not in valid_ids:
        st.session_state["selected_pid"] = valid_ids[0]

    if st.session_state["action_pid"] not in valid_ids:
        st.session_state["action_pid"] = valid_ids[0]


def sync_action_note(people_by_id):
    current_pid = st.session_state.get("action_pid")
    if not current_pid:
        st.session_state["action_note"] = ""
        st.session_state["action_note_pid"] = None
        return

    current_person = people_by_id[current_pid]
    if st.session_state.get("action_note_pid") != current_pid:
        st.session_state["action_note"] = current_person.get("note", "")
        st.session_state["action_note_pid"] = current_pid


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


def render_dashboard_content():
    db, registry = load_state()

    try:
        people = registry.all_people()
        ensure_valid_selection(people)
        people_by_id = {person["id"]: person for person in people}
        sync_action_note(people_by_id)

        faulty_count = sum(1 for person in people if person["faulty"])
        total_sightings = sum(person.get("sightings", 0) for person in people)

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Known Persons", len(people))
        metric2.metric("Faulty Persons", faulty_count)
        metric3.metric("Total Sightings", total_sightings)
        metric4.metric("Last Refresh", "Live")

        st.caption(
            f"Latest dashboard sync updates automatically from {DB_NAME}."
        )

        st.divider()
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Detected Persons")
            if not people:
                st.info("No persons stored yet. Start the main camera pipeline first.")
            else:
                options = [person["id"] for person in people]
                st.selectbox(
                    "Choose a person",
                    options=options,
                    key="selected_pid",
                )
                selected_person = people_by_id[st.session_state["selected_pid"]]
                show_person_card(selected_person)

        with right:
            st.subheader("Flag Person")
            if people:
                st.selectbox(
                    "Person ID",
                    options=[person["id"] for person in people],
                    key="action_pid",
                )
                current = people_by_id[st.session_state["action_pid"]]

                if st.session_state.get("action_note_pid") != current["id"]:
                    st.session_state["action_note"] = current.get("note", "")
                    st.session_state["action_note_pid"] = current["id"]

                st.text_area(
                    "Reason / Note",
                    key="action_note",
                    placeholder="Explain why this person is suspicious or faulty",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Mark Faulty", use_container_width=True):
                        registry.mark_faulty(
                            current["id"],
                            faulty=True,
                            note=st.session_state["action_note"].strip(),
                        )
                        st.success(f"{current['id']} marked as faulty.")
                        st.rerun()
                with col2:
                    if st.button("Clear Flag", use_container_width=True):
                        registry.mark_faulty(
                            current["id"],
                            faulty=False,
                            note=st.session_state["action_note"].strip(),
                        )
                        st.success(f"{current['id']} cleared.")
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
    finally:
        db.close()


@st.fragment(run_every=3)
def live_dashboard_fragment():
    if st.session_state.get("auto_refresh_enabled", True):
        render_dashboard_content()


def show_dashboard():
    init_session_state()

    st.title("Faulty Person Detection System")
    st.caption(
        "Review detected persons, mark anyone as faulty or suspicious, and monitor recent sightings."
    )

    control1, control2 = st.columns([1, 1])
    with control1:
        st.checkbox("Auto Refresh", key="auto_refresh_enabled")
    with control2:
        st.write("")
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()

    if st.session_state["auto_refresh_enabled"]:
        live_dashboard_fragment()
    else:
        render_dashboard_content()


if __name__ == "__main__":
    show_dashboard()

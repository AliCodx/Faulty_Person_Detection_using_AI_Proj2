import cv2
import numpy as np
import traceback

from cameras.camera_manager import CameraManager
from config import (
    AUTO_DISCOVER_CAMERAS,
    BUILTIN_CAMERA_INDEX,
    CAMERAS,
    CAMERA_BACKEND,
    CAMERA_EMPTY_FRAME_TOLERANCE,
    CAMERA_SCAN_MAX_INDEX,
    CAMERA_WARMUP_FRAMES,
    DB_NAME,
    DUPLICATE_MERGE_THRESHOLD,
    FAULTY_MATCH_BOOST,
    ID_SWITCH_MARGIN,
    MAX_ACTIVE_CAMERAS,
    PREFER_EXTERNAL_CAMERAS,
    RECENT_CROSS_CAMERA_SECONDS,
    RECENT_MATCH_BOOST,
    REID_THRESHOLD,
    TRACK_STICKINESS_THRESHOLD,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_MISSED,
    MODEL_NAME,
)
from core.system import TrackingSystem
from db.database import Database
from detection.yolo_detector import YoloDetector
from identity.registry import Registry
from reid.feature_extractor import FeatureExtractor
from reid.id_manager import IDManager
from ui.click_handler import ClickHandler


def render_message_window(title, lines):
    canvas = np.zeros((420, 920, 3), dtype=np.uint8)
    y = 50

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        y += 40

    cv2.namedWindow(title)

    while True:
        cv2.imshow(title, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def make_mouse_callback(click_handler, cam_id):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pid = click_handler.pick_from_point(cam_id, x, y)
            if pid:
                click_handler.mark_faulty(pid, note="Marked from live camera view")

    return on_mouse


def run():
    db = Database(DB_NAME)
    registry = Registry(db)

    detector = YoloDetector()
    extractor = FeatureExtractor(model_name=MODEL_NAME)
    id_manager = IDManager(
        registry,
        threshold=REID_THRESHOLD,
        track_stickiness_threshold=TRACK_STICKINESS_THRESHOLD,
        faulty_match_boost=FAULTY_MATCH_BOOST,
        recent_cross_camera_seconds=RECENT_CROSS_CAMERA_SECONDS,
        recent_match_boost=RECENT_MATCH_BOOST,
        id_switch_margin=ID_SWITCH_MARGIN,
        duplicate_merge_threshold=DUPLICATE_MERGE_THRESHOLD,
    )
    system = TrackingSystem(
        detector,
        extractor,
        id_manager,
        registry,
        db,
        tracker_iou_threshold=TRACKER_IOU_THRESHOLD,
        tracker_max_missed=TRACKER_MAX_MISSED,
    )
    click_handler = ClickHandler(registry)

    cams = CameraManager(
        CAMERAS,
        backend=CAMERA_BACKEND,
        warmup_frames=CAMERA_WARMUP_FRAMES,
        auto_discover=AUTO_DISCOVER_CAMERAS,
        scan_max_index=CAMERA_SCAN_MAX_INDEX,
        max_active_cameras=MAX_ACTIVE_CAMERAS,
        prefer_external_cameras=PREFER_EXTERNAL_CAMERAS,
        builtin_camera_index=BUILTIN_CAMERA_INDEX,
    )
    active_sources = cams.active_sources()

    if cams.failed_sources:
        print(f"[WARN] Could not open configured camera source(s): {cams.failed_sources}")

    if active_sources:
        print("[INFO] Active camera sources:")
        for source in active_sources:
            print(f"  - source {source} via {cams.backend_for_source(source)}")
        if (
            PREFER_EXTERNAL_CAMERAS
            and BUILTIN_CAMERA_INDEX not in active_sources
            and len(active_sources) == MAX_ACTIVE_CAMERAS
        ):
            print(
                f"[INFO] Using up to {MAX_ACTIVE_CAMERAS} cameras and skipping built-in camera "
                f"index {BUILTIN_CAMERA_INDEX} when enough other cameras are available."
            )

    if not active_sources:
        render_message_window(
            "Camera Error",
            [
                "No camera source could be opened.",
                "Press Q or ESC to close this window.",
                "Check Windows camera permission and close Zoom/Teams/browser tabs.",
                "If you use CCTV/IP cameras, replace CAMERAS indexes with RTSP URLs.",
                "You can also run: python camera_test.py",
            ],
        )
        return 1

    for cam_id in active_sources:
        cv2.namedWindow(f"Camera {cam_id}")
        cv2.setMouseCallback(f"Camera {cam_id}", make_mouse_callback(click_handler, cam_id))

    empty_frame_count = 0

    while True:
        frames = cams.read_frames()

        if not frames:
            empty_frame_count += 1

            if empty_frame_count == 1:
                print("[WARN] Camera opened, but frames are not ready yet. Waiting...")

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

            if empty_frame_count >= CAMERA_EMPTY_FRAME_TOLERANCE:
                render_message_window(
                    "Camera Warning",
                    [
                        "Camera opened, but no frames were received.",
                        "Press Q or ESC to close this window.",
                        "The camera may be busy in another app or blocked by permissions.",
                    ],
                )
                break

            continue

        empty_frame_count = 0

        for cam_id, frame in frames:
            try:
                detections = system.process(frame, cam_id)
            except Exception as exc:
                cams.release()
                cv2.destroyAllWindows()
                render_message_window(
                    "Processing Error",
                    [
                        f"Processing failed for camera {cam_id}.",
                        f"Error: {exc}",
                        "Press Q or ESC to close this window.",
                    ],
                )
                raise

            click_handler.update_detections(cam_id, detections)

            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                pid = detection["pid"]

                color = (0, 0, 255) if detection["faulty"] else (0, 255, 0)
                label = f"{pid} | {'FAULTY' if detection['faulty'] else 'Normal'}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                "Click a person to mark as faulty",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(f"Camera {cam_id}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cams.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception:
        print("[ERROR] Unhandled exception in main.py")
        print(traceback.format_exc())
        render_message_window(
            "Application Error",
            [
                "The application stopped because of an unexpected error.",
                "Check the terminal output for the full traceback.",
                "Press Q or ESC to close this window.",
            ],
        )
        raise

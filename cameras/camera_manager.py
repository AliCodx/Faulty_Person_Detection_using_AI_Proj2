# cameras/camera_manager.py

import cv2


class CameraManager:
    def __init__(
        self,
        sources,
        backend="dshow",
        warmup_frames=10,
        auto_discover=True,
        scan_max_index=5,
        max_active_cameras=None,
        prefer_external_cameras=True,
        builtin_camera_index=0,
    ):
        self.configured_sources = list(sources)
        self.caps = {}
        self.failed_sources = []
        self.backend = backend
        self.warmup_frames = warmup_frames
        self.auto_discover = auto_discover
        self.scan_max_index = scan_max_index
        self.opened_backends = {}
        self.max_active_cameras = max_active_cameras
        self.prefer_external_cameras = prefer_external_cameras
        self.builtin_camera_index = builtin_camera_index

        for source in sources:
            cap, backend_used = self._open_capture(source)
            if cap is not None and cap.isOpened():
                self.caps[source] = cap
                self.opened_backends[source] = backend_used
                self._warmup_capture(cap)
            else:
                self.failed_sources.append(source)
                if cap is not None:
                    cap.release()

        if self.auto_discover and self._should_auto_discover():
            self._auto_discover_sources()

        self._apply_source_selection()

    def _should_auto_discover(self):
        if not self.configured_sources:
            return True

        return any(isinstance(source, int) for source in self.configured_sources)

    def _backend_candidates(self, source):
        if not isinstance(source, int):
            return [("ANY", cv2.CAP_ANY)]

        names = []
        preferred = str(self.backend).upper()
        if preferred == "DSHOW":
            names.append(("DSHOW", cv2.CAP_DSHOW))
        elif preferred == "MSMF":
            names.append(("MSMF", cv2.CAP_MSMF))
        elif preferred == "ANY":
            names.append(("ANY", cv2.CAP_ANY))

        for name, flag in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
            if name not in {candidate[0] for candidate in names}:
                names.append((name, flag))

        return names

    def _open_capture(self, source):
        for backend_name, backend_flag in self._backend_candidates(source):
            cap = cv2.VideoCapture(source, backend_flag)
            if cap.isOpened():
                return cap, backend_name
            cap.release()
        return None, None

    def _auto_discover_sources(self):
        for source in range(int(self.scan_max_index) + 1):
            if source in self.caps:
                continue

            cap, backend_used = self._open_capture(source)
            if cap is not None and cap.isOpened():
                self.caps[source] = cap
                self.opened_backends[source] = backend_used
                self._warmup_capture(cap)

    def _apply_source_selection(self):
        active_sources = list(self.caps.keys())

        if not active_sources:
            return

        if self.max_active_cameras is None or self.max_active_cameras <= 0:
            return

        if len(active_sources) <= self.max_active_cameras:
            return

        selected_sources = self._select_sources(active_sources)
        sources_to_drop = [source for source in active_sources if source not in selected_sources]

        for source in sources_to_drop:
            self.caps[source].release()
            del self.caps[source]
            self.opened_backends.pop(source, None)

    def _select_sources(self, active_sources):
        integer_sources = sorted(source for source in active_sources if isinstance(source, int))
        non_integer_sources = [source for source in active_sources if not isinstance(source, int)]

        if non_integer_sources:
            ordered_sources = non_integer_sources + integer_sources
            return ordered_sources[: self.max_active_cameras]

        if (
            self.prefer_external_cameras
            and len(integer_sources) > self.max_active_cameras
            and self.builtin_camera_index in integer_sources
        ):
            filtered_sources = [
                source for source in integer_sources if source != self.builtin_camera_index
            ]
            if len(filtered_sources) >= self.max_active_cameras:
                return filtered_sources[: self.max_active_cameras]

        return integer_sources[: self.max_active_cameras]

    def _warmup_capture(self, cap):
        for _ in range(max(0, int(self.warmup_frames))):
            cap.read()

    def read_frames(self):
        frames = []

        for source, cap in self.caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append((source, frame))

        return frames

    def active_sources(self):
        ordered = list(self.caps.keys())
        integer_sources = sorted(source for source in ordered if isinstance(source, int))
        non_integer_sources = [source for source in ordered if not isinstance(source, int)]
        return non_integer_sources + integer_sources

    def backend_for_source(self, source):
        return self.opened_backends.get(source, "UNKNOWN")

    def release(self):
        for cap in self.caps.values():
            cap.release()

class ClickHandler:
    def __init__(self, registry):
        self.registry = registry
        self.selected = None
        self.live_detections = {}

    def update_detections(self, cam_id, detections):
        self.live_detections[cam_id] = detections

    def pick_from_point(self, cam_id, x, y):
        for detection in self.live_detections.get(cam_id, []):
            x1, y1, x2, y2 = detection["box"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected = detection["pid"]
                return self.selected
        return None

    def mark_faulty(self, pid=None, note=None):
        target = pid or self.selected
        if target:
            self.registry.mark_faulty(target, faulty=True, note=note)
        return target

    def clear_faulty(self, pid=None, note=None):
        target = pid or self.selected
        if target:
            self.registry.mark_faulty(target, faulty=False, note=note)
        return target

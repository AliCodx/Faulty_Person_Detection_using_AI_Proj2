# detection/yolo_detector.py

from ultralytics import YOLO

class YoloDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame)
        boxes = []

        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) == 0:  # person class
                    boxes.append(b.xyxy[0].tolist())

        return boxes
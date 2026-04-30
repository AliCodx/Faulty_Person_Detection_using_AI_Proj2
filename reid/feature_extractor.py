import cv2
import numpy as np


class FeatureExtractor:
    """
    Lightweight appearance descriptor for person re-identification.
    This is not as strong as a dedicated ReID network, but it is stable,
    deterministic, and works without extra model downloads.
    """

    def __init__(self, output_size=(64, 128)):
        self.output_size = output_size

    def extract(self, image):
        if image is None or image.size == 0:
            return np.zeros(192, dtype=np.float32)

        resized = cv2.resize(image, self.output_size)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        color_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        texture = cv2.resize(gray, (8, 8)).astype(np.float32).flatten() / 255.0
        channel_stats = resized.reshape(-1, 3).astype(np.float32)
        means = channel_stats.mean(axis=0) / 255.0
        stds = channel_stats.std(axis=0) / 255.0

        feature = np.concatenate([color_hist, texture, means, stds]).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        return feature

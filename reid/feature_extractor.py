import cv2
import numpy as np
import torch
import torchreid


class FeatureExtractor:
    """
    Person re-identification embedding extractor.
    Uses OSNet with locally cached pretrained weights when available,
    and falls back to a lightweight handcrafted descriptor otherwise.
    """

    def __init__(self, model_name="osnet_x1_0", image_size=(128, 256), device="cpu"):
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device(device)
        self.model = None
        self.output_dim = 512
        self.fallback_dim = 198
        self.use_deep_features = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        try:
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1000,
                pretrained=True,
                use_gpu=False,
            )
            self.model.eval()
            self.use_deep_features = True
        except Exception as exc:
            print(f"[WARN] Falling back to lightweight ReID features: {exc}")
            self.model = None
            self.output_dim = self.fallback_dim

    def _normalize(self, feature):
        feature = np.asarray(feature, dtype=np.float32).flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

    def _extract_lightweight(self, image):
        if image is None or image.size == 0:
            return np.zeros(self.fallback_dim, dtype=np.float32)

        resized = cv2.resize(image, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        color_hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 4, 4],
            [0, 180, 0, 256, 0, 256],
        )
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        texture = cv2.resize(gray, (8, 8)).astype(np.float32).flatten() / 255.0
        channel_stats = resized.reshape(-1, 3).astype(np.float32)
        means = channel_stats.mean(axis=0) / 255.0
        stds = channel_stats.std(axis=0) / 255.0

        feature = np.concatenate([color_hist, texture, means, stds]).astype(np.float32)
        return self._normalize(feature)

    def _preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.image_size)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)

    def extract(self, image):
        if image is None or image.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        if not self.use_deep_features or self.model is None:
            return self._extract_lightweight(image)

        try:
            tensor = self._preprocess(image).to(self.device)
            with torch.inference_mode():
                embedding = self.model(tensor)

            if isinstance(embedding, (list, tuple)):
                embedding = embedding[0]

            feature = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return self._normalize(feature)
        except Exception as exc:
            print(f"[WARN] Deep ReID extraction failed, using fallback features: {exc}")
            self.use_deep_features = False
            self.output_dim = self.fallback_dim
            return self._extract_lightweight(image)

import os
import numpy as np
from ultralytics import YOLO


class PestDetector:
    """
    YOLO-based pest detection module.
    Lazy loads model for isolation.
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.5
    ):

        base_dir = os.path.dirname(os.path.dirname(__file__))

        if model_path is None:
            model_path = os.path.join(
                base_dir, "models", "pest_detection.pt"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pest detector model not found at {model_path}"
            )

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        self.model = None  # Lazy load

    # -------------------------------------------------
    # Lazy Model Loader
    # -------------------------------------------------
    def _load_model(self):
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                raise RuntimeError(f"YOLO model loading failed: {str(e)}")

    # -------------------------------------------------
    # Detection
    # -------------------------------------------------
    def detect(self, image_bgr: np.ndarray):

        # Attempt to load model
        try:
            self._load_model()
        except Exception as e:
            return {
                "error": str(e)
            }

        try:
            results = self.model.predict(
                source=image_bgr,
                conf=self.conf_threshold,
                verbose=False
            )

            detections = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    class_name = self.model.names[class_id]

                    detections.append({
                        "label": class_name,
                        "confidence": confidence,
                        "bbox": (
                            int(x1), int(y1),
                            int(x2), int(y2)
                        )
                    })

            return {
                "pest_count": len(detections),
                "pests": detections
            }

        except Exception as e:
            return {
                "error": f"Pest detection failed: {str(e)}"
            }

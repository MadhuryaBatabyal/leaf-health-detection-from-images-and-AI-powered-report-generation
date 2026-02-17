import os
import numpy as np
from ultralytics import YOLO


class LeafDetector:
    """
    YOLO-based leaf detection module.
    Lazy loads model to avoid crashing pipeline initialization.
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.5
    ):

        base_dir = os.path.dirname(os.path.dirname(__file__))

        if model_path is None:
            model_path = os.path.join(
                base_dir, "models", "leaf_detection.pt"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Leaf detector model not found at {model_path}"
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
                raise RuntimeError(f"Leaf YOLO model loading failed: {str(e)}")

    # -------------------------------------------------
    # Detection
    # -------------------------------------------------
    def detect(
        self,
        image_bgr: np.ndarray,
        return_multiple: bool = False
    ):

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

            boxes = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((
                        int(x1), int(y1),
                        int(x2), int(y2)
                    ))

            if not boxes:
                return None if not return_multiple else []

            if return_multiple:
                return boxes

            # Return largest leaf
            largest_box = max(
                boxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )

            return largest_box

        except Exception as e:
            return {
                "error": f"Leaf detection failed: {str(e)}"
            }

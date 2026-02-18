import os
import json
import numpy as np
import cv2
import onnxruntime as ort


class LeafDiseasePredictor:
    """
    ONNX-based leaf disease classification module.
    """

    def __init__(self, model_path=None, labels_path=None):

        base_dir = os.path.dirname(os.path.dirname(__file__))

        if model_path is None:
            model_path = os.path.join(base_dir, "models", "leaf_disease.onnx")

        if labels_path is None:
            labels_path = os.path.join(base_dir, "leaf_labels.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        self.model_path = model_path
        self.labels_path = labels_path

        self.session = None
        self.class_names = None

    # -------------------------------------------------
    # Lazy Model Loader
    # -------------------------------------------------
    def _load_model(self):
        if self.session is None:
            try:
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=["CPUExecutionProvider"]
                )
                with open(self.labels_path, "r") as f:
                    self.class_names = json.load(f)

            except Exception as e:
                raise RuntimeError(f"ONNX model load failed: {str(e)}")

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, image_bgr: np.ndarray):

        try:
            self._load_model()
        except Exception as e:
            return {"error": str(e)}

        try:
            # Convert BGR → RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Resize to model input
            image_resized = cv2.resize(image_rgb, (224, 224))

            # Convert to float32
            image_resized = image_resized.astype(np.float32)

            # MobileNetV2 normalization
            image_resized = (image_resized / 127.5) - 1.0

            # Add batch dimension
            image_batch = np.expand_dims(image_resized, axis=0)

            # Run ONNX inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: image_batch})

            preds = outputs[0][0]

            class_id = int(np.argmax(preds))
            confidence = float(preds[class_id])

            return {
                "class_id": class_id,
                "label": self.class_names[class_id],
                "confidence": confidence,
                "all_probabilities": preds.tolist()
            }

        except Exception as e:
            return {"error": f"ONNX inference failed: {str(e)}"}

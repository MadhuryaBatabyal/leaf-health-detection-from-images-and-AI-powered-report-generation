import traceback

from inference.leaf_detection import LeafDetector
from inference.leaf_disease import LeafDiseasePredictor
from inference.pest_detection import PestDetector
from inference.dryness_estimator import LeafDrynessAnalyzer
from inference.green_index import GreenIndexCalculator


class IsolatedPlantAnalyzer:
    """
    Fully isolated modular pipeline.
    Each module runs independently.
    Failures in one module do NOT affect others.
    """

    def __init__(self):

        self.leaf_detector = LeafDetector()
        self.disease_predictor = LeafDiseasePredictor()
        self.pest_detector = PestDetector()
        self.dryness_analyzer = LeafDrynessAnalyzer()
        self.green_index_analyzer = GreenIndexCalculator()

    # -------------------------------------------------
    # Safe Module Runner
    # -------------------------------------------------
    def _run_safe(self, func, *args):
        try:
            return func(*args)
        except Exception as e:
            return {
                "error": str(e),
                "trace": traceback.format_exc()
            }

    # -------------------------------------------------
    # Main Analysis
    # -------------------------------------------------
    def analyze(self, image):

        results = {}

        # ---------------------------------------------
        # 1️⃣ Leaf Detection
        # ---------------------------------------------
        leaf_bbox = self._run_safe(self.leaf_detector.detect, image)
        results["leaf_detection"] = leaf_bbox

        # Crop leaf if detected
        cropped_leaf = image
        if isinstance(leaf_bbox, tuple) and len(leaf_bbox) == 4:
            x1, y1, x2, y2 = leaf_bbox
            cropped_leaf = image[y1:y2, x1:x2]

        # ---------------------------------------------
        # 2️⃣ Disease Detection
        # ---------------------------------------------
        results["disease"] = self._run_safe(
            self.disease_predictor.predict,
            cropped_leaf
        )

        # ---------------------------------------------
        # 3️⃣ Pest Detection
        # ---------------------------------------------
        results["pests"] = self._run_safe(
            self.pest_detector.detect,
            cropped_leaf
        )

        # ---------------------------------------------
        # 4️⃣ Dryness Analysis
        # ---------------------------------------------
        results["dryness"] = self._run_safe(
            self.dryness_analyzer.analyze,
            cropped_leaf
        )

        # ---------------------------------------------
        # 5️⃣ Green Index
        # ---------------------------------------------
        results["green_index"] = self._run_safe(
            self.green_index_analyzer.compute,
            cropped_leaf
        )

        return results

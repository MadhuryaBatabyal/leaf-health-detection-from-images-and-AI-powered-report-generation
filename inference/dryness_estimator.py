import numpy as np
import cv2


class LeafDrynessAnalyzer:
    """
    Computes multi-metric dryness analysis for cropped leaf image.
    """

    def __init__(self):
        pass

    def analyze(self, image_bgr: np.ndarray):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        image_rgb = image_rgb.astype(np.float32)
        image_hsv = image_hsv.astype(np.float32)

        # --- 1️⃣ Green Ratio ---
        green_channel = image_rgb[:, :, 1]
        green_ratio = float(np.mean(green_channel) / 255.0)

        # --- 2️⃣ Brown Ratio ---
        brown_mask = cv2.inRange(
            image_hsv.astype(np.uint8),
            (10, 40, 40),
            (30, 255, 255)
        )
        brown_ratio = float(np.sum(brown_mask > 0) / brown_mask.size)

        # --- 3️⃣ Saturation Mean ---
        saturation = image_hsv[:, :, 1] / 255.0
        saturation_mean = float(np.mean(saturation))

        # --- 4️⃣ Texture Variance (Laplacian) ---
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize texture variance roughly
        texture_score = min(laplacian_var / 100.0, 1.0)

        # --- Combined Dryness Score ---
        dryness_score = (
            0.35 * brown_ratio +
            0.25 * (1 - green_ratio) +
            0.20 * (1 - saturation_mean) +
            0.20 * texture_score
        )

        dryness_score = float(np.clip(dryness_score, 0, 1))

        # --- Dryness Level ---
        if dryness_score < 0.3:
            level = "Low"
        elif dryness_score < 0.6:
            level = "Moderate"
        else:
            level = "High"

        return {
            "dryness_score": dryness_score,
            "dryness_level": level,
            "metrics": {
                "green_ratio": green_ratio,
                "brown_ratio": brown_ratio,
                "saturation_mean": saturation_mean,
                "texture_variance": laplacian_var
            }
        }

import numpy as np
import cv2


class GreenIndexCalculator:
    """
    Computes green index metrics from cropped leaf image.
    """

    def __init__(self):
        pass

    def compute(self, image_bgr: np.ndarray):
        """
        Args:
            image_bgr: cropped leaf image

        Returns:
            dict containing:
                excess_green
                normalized_green
                green_score (0-1)
        """

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32)

        R = image_rgb[:, :, 0]
        G = image_rgb[:, :, 1]
        B = image_rgb[:, :, 2]

        # Excess Green
        exg = 2 * G - R - B
        exg_mean = float(np.mean(exg))

        # Normalized Green
        denominator = (R + G + B) + 1e-6
        normalized_green = G / denominator
        normalized_green_mean = float(np.mean(normalized_green))

        # Combine into 0-1 green score
        green_score = float(np.clip(normalized_green_mean, 0, 1))

        return {
            "excess_green": exg_mean,
            "normalized_green": normalized_green_mean,
            "green_score": green_score
        }

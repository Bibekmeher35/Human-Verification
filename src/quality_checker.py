from __future__ import annotations

import cv2
import numpy as np


def check_quality(image_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    height, width = image_bgr.shape[:2]

    return {
        "width": int(width),
        "height": int(height),
        "blur_score": round(blur_score, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
    }

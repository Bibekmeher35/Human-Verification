from __future__ import annotations

import numpy as np


def check_face_crop_geometry(image_bgr: np.ndarray, primary_face: dict | None) -> dict:
    h, w = image_bgr.shape[:2]
    if not primary_face:
        return {
            "face_height_ratio": None,
            "face_width_ratio": None,
            "space_below_face_ratio": None,
            "geometry_status": "no_face",
        }

    x1, y1, x2, y2 = primary_face["box"]
    face_w = max(0.0, x2 - x1)
    face_h = max(0.0, y2 - y1)
    face_height_ratio = face_h / h
    face_width_ratio = face_w / w
    space_below_face_ratio = max(0.0, h - y2) / h
    face_center_x_ratio = ((x1 + x2) / 2.0) / w
    face_center_y_ratio = ((y1 + y2) / 2.0) / h

    return {
        "face_height_ratio": round(float(face_height_ratio), 4),
        "face_width_ratio": round(float(face_width_ratio), 4),
        "space_below_face_ratio": round(float(space_below_face_ratio), 4),
        "face_center_x_ratio": round(float(face_center_x_ratio), 4),
        "face_center_y_ratio": round(float(face_center_y_ratio), 4),
        "geometry_status": "computed",
    }

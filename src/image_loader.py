from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def load_image_bgr(image_path: str | Path, max_side: int = 1600) -> np.ndarray:
    """Load JPG/JPEG/PNG/HEIC as OpenCV BGR image.

    Uses Pillow first so HEIC and EXIF orientation are handled correctly.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        width, height = img.size
        largest_side = max(width, height)
        if largest_side > max_side:
            scale = max_side / float(largest_side)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        rgb = np.array(img)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def image_size(image_bgr: np.ndarray) -> Tuple[int, int]:
    h, w = image_bgr.shape[:2]
    return w, h

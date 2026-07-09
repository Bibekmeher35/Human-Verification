"""Lightweight AI generated/synthetic image suspicion detector using MobileNetV3 TFLite model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class MobileNetV3AIDetector:
    """Runs inference on a MobileNetV3 TFLite model to detect synthetic/AI images."""

    def __init__(self, model_path: str, input_size: tuple[int, int] = (224, 224)) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.interpreter = None
        self.enabled = False

        if not self.model_path.exists():
            print(
                f"\033[93m[Warning] AI suspicion model not found at {self.model_path}.\n"
                f"To enable AI generated image detection, train and convert the model first.\n"
                f"Skipping AI suspicion check for now.\033[0m"
            )
            return

        # Attempt to load TFLite interpreter dynamically
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        except ImportError:
            try:
                import tensorflow.lite as tflite
                self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            except ImportError:
                print(
                    "\033[93m[Warning] Neither 'tflite-runtime' nor 'tensorflow' packages are installed.\n"
                    "Please install one of them to run AI generated image validation.\n"
                    "Skipping AI suspicion check for now.\033[0m"
                )
                return

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.enabled = True

    def detect(self, image_bgr: np.ndarray) -> float | None:
        """Returns the synthetic suspicion score between 0.0 and 1.0.
        
        Returns None if the detector is disabled/model is not loaded.
        """
        if not self.enabled or self.interpreter is None:
            return None

        # Preprocess: BGR -> RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input shape
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Expand batch dimension and convert to float32
        # Standard MobileNetV3 Small model we trained has scaling layer inside,
        # so it expects float32 values in range [0.0, 255.0]
        input_data = np.expand_dims(resized.astype(np.float32), axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Get prediction (sigmoidal probability output)
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        suspicion_score = float(output_data[0][0])

        return suspicion_score

    def close(self) -> None:
        """Release interpreter resources if any."""
        # TFLite interpreter does not strictly require closing, but we clean up references
        self.interpreter = None
        self.enabled = False

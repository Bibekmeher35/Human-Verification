from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ValidatorConfig
from .ai_detector import MobileNetV3AIDetector
from .body_crop_checker import check_face_crop_geometry
from .body_pose_checker import MediaPipeBodyChecker
from .decision_engine import decide
from .face_detector_scrfd import SCRFDFaceDetector
from .image_loader import load_image_bgr
from .quality_checker import check_quality


class PersonPhotoValidator:
    def __init__(self, cfg: ValidatorConfig | None = None) -> None:
        self.cfg = cfg or ValidatorConfig()
        self.face_detector = SCRFDFaceDetector(
            model_path=self.cfg.scrfd_model_path,
            input_size=self.cfg.scrfd_input_size,
            det_threshold=self.cfg.scrfd_det_threshold,
            nms_threshold=self.cfg.scrfd_nms_threshold,
        )
        self.body_checker = MediaPipeBodyChecker(
            pose_min_visibility=self.cfg.pose_min_visibility,
            shoulder_min_visibility=self.cfg.shoulder_min_visibility,
            torso_min_visibility=self.cfg.torso_min_visibility,
            min_lowest_body_y_for_upper_body=self.cfg.min_lowest_body_y_for_upper_body,
            min_shoulder_to_hip_span=self.cfg.min_shoulder_to_hip_span,
            min_face_to_torso_span=self.cfg.min_face_to_torso_span,
        )
        self.ai_detector = MobileNetV3AIDetector(
            model_path=self.cfg.ai_detector_model_path,
            input_size=self.cfg.ai_detector_input_size,
        )

    def validate(self, image_path: str | Path) -> dict[str, Any]:
        image = load_image_bgr(image_path, max_side=self.cfg.max_image_side)
        quality = check_quality(image)
        face_result = self.face_detector.detect(image)
        primary_face = face_result["faces"][0] if face_result.get("faces") else None
        geometry = check_face_crop_geometry(image, primary_face)
        body = self.body_checker.check(image, primary_face)
        
        # Run AI generated suspicion check
        ai_suspicion_score = self.ai_detector.detect(image)
        
        result = decide(self.cfg, face_result, quality, geometry, body, ai_suspicion_score)
        result["image_path"] = str(image_path)
        result["debug"] = {
            "quality": quality,
            "face_result": face_result,
            "geometry": geometry,
            "body": body,
            "ai_suspicion_score": ai_suspicion_score,
        }
        return result

    def close(self) -> None:
        self.body_checker.close()
        self.ai_detector.close()

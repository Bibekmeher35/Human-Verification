from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidatorConfig:
    # Model
    scrfd_model_path: str = str(Path("models") / "scrfd_2.5g_bnkps.onnx")
    scrfd_input_size: tuple[int, int] = (640, 640)
    scrfd_det_threshold: float = 0.50
    scrfd_nms_threshold: float = 0.40
    face_confidence_min: float = 0.65

    # Image loading
    max_image_side: int = 1600
    min_image_width: int = 300
    min_image_height: int = 300

    # Quality thresholds
    blur_reject_threshold: float = 60.0
    blur_manual_threshold: float = 100.0
    brightness_min: float = 40.0
    brightness_max: float = 220.0
    contrast_min: float = 20.0

    # Face crop thresholds
    max_face_height_ratio: float = 0.45
    min_face_height_ratio: float = 0.06
    min_space_below_face_reject: float = 0.30
    min_space_below_face_manual: float = 0.45

    # Pose/body thresholds
    pose_min_visibility: float = 0.50
    shoulder_min_visibility: float = 0.50
    torso_min_visibility: float = 0.45
    # If the lowest reliable body landmark is below this fraction of image height,
    # the image likely contains more than head and shoulders.
    min_lowest_body_y_for_upper_body: float = 0.55
    min_shoulder_to_hip_span: float = 0.18
    min_face_to_torso_span: float = 0.32

    # AI Detector Config
    ai_detector_model_path: str = str(Path("models") / "ai_detector_mobilenetv3.tflite")
    ai_detector_input_size: tuple[int, int] = (224, 224)
    ai_suspicion_threshold_manual: float = 0.50
    ai_suspicion_threshold_strict: float = 0.75
    ai_suspicion_threshold_reject: float = 0.90

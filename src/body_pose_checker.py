from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeBodyChecker:
    """Uses MediaPipe Pose to verify whether more than head/shoulders are visible."""

    def __init__(
        self,
        pose_min_visibility: float = 0.50,
        shoulder_min_visibility: float = 0.50,
        torso_min_visibility: float = 0.45,
        min_lowest_body_y_for_upper_body: float = 0.55,
        min_shoulder_to_hip_span: float = 0.18,
        min_face_to_torso_span: float = 0.32,
    ) -> None:
        self.pose_min_visibility = pose_min_visibility
        self.shoulder_min_visibility = shoulder_min_visibility
        self.torso_min_visibility = torso_min_visibility
        self.min_lowest_body_y_for_upper_body = min_lowest_body_y_for_upper_body
        self.min_shoulder_to_hip_span = min_shoulder_to_hip_span
        self.min_face_to_torso_span = min_face_to_torso_span

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def check(self, image_bgr: np.ndarray, primary_face: dict[str, Any] | None) -> dict[str, Any]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return {
                "pose_found": False,
                "body_status": "pose_not_found",
                "body_confidence": 0.0,
                "reason": "No body pose landmarks found",
            }

        landmarks = result.pose_landmarks.landmark
        L = self.mp_pose.PoseLandmark

        def lm(name: Any) -> dict[str, float]:
            p = landmarks[name.value]
            return {"x": float(p.x), "y": float(p.y), "visibility": float(p.visibility)}

        points = {
            "nose": lm(L.NOSE),
            "left_shoulder": lm(L.LEFT_SHOULDER),
            "right_shoulder": lm(L.RIGHT_SHOULDER),
            "left_elbow": lm(L.LEFT_ELBOW),
            "right_elbow": lm(L.RIGHT_ELBOW),
            "left_hip": lm(L.LEFT_HIP),
            "right_hip": lm(L.RIGHT_HIP),
        }

        left_shoulder_visible = points["left_shoulder"]["visibility"] >= self.shoulder_min_visibility
        right_shoulder_visible = points["right_shoulder"]["visibility"] >= self.shoulder_min_visibility
        shoulders_visible = left_shoulder_visible and right_shoulder_visible

        left_hip_visible = points["left_hip"]["visibility"] >= self.torso_min_visibility
        right_hip_visible = points["right_hip"]["visibility"] >= self.torso_min_visibility
        hips_visible = left_hip_visible or right_hip_visible

        left_elbow_visible = points["left_elbow"]["visibility"] >= self.torso_min_visibility
        right_elbow_visible = points["right_elbow"]["visibility"] >= self.torso_min_visibility
        elbows_visible = left_elbow_visible or right_elbow_visible

        reliable = [p for p in points.values() if p["visibility"] >= self.pose_min_visibility]
        lowest_reliable_y = max([p["y"] for p in reliable], default=0.0)

        shoulder_y_values = []
        if left_shoulder_visible:
            shoulder_y_values.append(points["left_shoulder"]["y"])
        if right_shoulder_visible:
            shoulder_y_values.append(points["right_shoulder"]["y"])
        avg_shoulder_y = float(np.mean(shoulder_y_values)) if shoulder_y_values else None

        hip_y_values = []
        if left_hip_visible:
            hip_y_values.append(points["left_hip"]["y"])
        if right_hip_visible:
            hip_y_values.append(points["right_hip"]["y"])
        avg_hip_y = float(np.mean(hip_y_values)) if hip_y_values else None

        shoulder_to_hip_span = None
        if avg_shoulder_y is not None and avg_hip_y is not None:
            shoulder_to_hip_span = avg_hip_y - avg_shoulder_y

        face_to_lowest_span = None
        if primary_face:
            _, fy1, _, fy2 = primary_face["box"]
            face_bottom_ratio = fy2 / h
            face_to_lowest_span = lowest_reliable_y - face_bottom_ratio

        # Decision for body visibility.
        # Strong accept: shoulders + hips/elbows or reliable body landmarks extend lower in the image.
        has_torso_signal = (
            hips_visible
            or elbows_visible
            or lowest_reliable_y >= self.min_lowest_body_y_for_upper_body
            or (shoulder_to_hip_span is not None and shoulder_to_hip_span >= self.min_shoulder_to_hip_span)
            or (face_to_lowest_span is not None and face_to_lowest_span >= self.min_face_to_torso_span)
        )

        if shoulders_visible and has_torso_signal:
            status = "more_than_head_shoulders"
            confidence = 0.90
            reason = "Shoulders and lower upper-body landmarks are visible"
        elif shoulders_visible:
            status = "head_shoulders_only"
            confidence = 0.70
            reason = "Shoulders visible, but torso/lower body is not clear"
        elif has_torso_signal:
            status = "manual_body_partial"
            confidence = 0.55
            reason = "Some lower body landmarks are visible, but shoulders are not reliable"
        else:
            status = "face_or_head_only"
            confidence = 0.40
            reason = "Body landmarks do not show more than head/shoulders"

        return {
            "pose_found": True,
            "body_status": status,
            "body_confidence": confidence,
            "reason": reason,
            "landmark_summary": {
                "shoulders_visible": shoulders_visible,
                "hips_visible": hips_visible,
                "elbows_visible": elbows_visible,
                "lowest_reliable_y_ratio": round(float(lowest_reliable_y), 4),
                "avg_shoulder_y_ratio": round(avg_shoulder_y, 4) if avg_shoulder_y is not None else None,
                "avg_hip_y_ratio": round(avg_hip_y, 4) if avg_hip_y is not None else None,
                "shoulder_to_hip_span_ratio": round(shoulder_to_hip_span, 4) if shoulder_to_hip_span is not None else None,
                "face_to_lowest_span_ratio": round(face_to_lowest_span, 4) if face_to_lowest_span is not None else None,
            },
        }

    def close(self) -> None:
        self.pose.close()

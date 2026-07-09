from __future__ import annotations

from config import ValidatorConfig


def decide(
    cfg: ValidatorConfig,
    face_result: dict,
    quality: dict,
    geometry: dict,
    body: dict,
    ai_suspicion_score: float | None = None,
) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []

    width = quality["width"]
    height = quality["height"]
    blur_score = quality["blur_score"]
    brightness = quality["brightness"]
    contrast = quality["contrast"]

    scores = {
        "image_width": width,
        "image_height": height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "face_count": face_result.get("face_count", 0),
        "body_status": body.get("body_status"),
        "body_confidence": body.get("body_confidence"),
        **{k: v for k, v in geometry.items() if k.endswith("ratio")},
    }

    if ai_suspicion_score is not None:
        scores["ai_suspicion_score"] = round(ai_suspicion_score, 4)

    face_count = face_result.get("face_count", 0)
    if face_count == 0:
        return _result("rejected", ["No face/person detected"], warnings, scores)

    # Evaluate AI generated suspicion
    ai_flag = None
    ai_reasons = []
    if ai_suspicion_score is not None:
        if ai_suspicion_score >= cfg.ai_suspicion_threshold_reject:
            ai_flag = "rejected"
            ai_reasons.append("Highly suspicious synthetic/AI-generated image")
        elif ai_suspicion_score >= cfg.ai_suspicion_threshold_strict:
            ai_flag = "manual_verification"
            warnings.append("High AI-generated image suspicion")
        elif ai_suspicion_score >= cfg.ai_suspicion_threshold_manual:
            ai_flag = "manual_verification"
            warnings.append("Medium AI-generated image suspicion")

    if ai_flag == "rejected":
        return _result("rejected", ai_reasons, warnings, scores)

    faces = face_result.get("faces", [])
    primary_face = faces[0]
    face_confidence = primary_face["confidence"]
    scores["face_confidence"] = face_confidence

    if face_count > 1:
        # Group images should generally not pass automatically.
        return _result("manual_verification", ["Multiple faces detected"], warnings, scores)

    if face_confidence < cfg.face_confidence_min:
        return _result("manual_verification", ["Face confidence is low"], warnings, scores)

    if width < cfg.min_image_width or height < cfg.min_image_height:
        return _result("manual_verification", ["Image resolution is too small"], warnings, scores)

    if blur_score < cfg.blur_reject_threshold:
        return _result("rejected", ["Image is too blurry"], warnings, scores)
    if blur_score < cfg.blur_manual_threshold:
        warnings.append("Image is slightly blurry")

    if brightness < cfg.brightness_min:
        return _result("manual_verification", ["Image is too dark"], warnings, scores)
    if brightness > cfg.brightness_max:
        return _result("manual_verification", ["Image is too bright"], warnings, scores)
    if contrast < cfg.contrast_min:
        warnings.append("Image contrast is low")

    face_height_ratio = geometry.get("face_height_ratio")
    space_below_face_ratio = geometry.get("space_below_face_ratio")

    if face_height_ratio is not None:
        if face_height_ratio > cfg.max_face_height_ratio:
            return _result("rejected", ["Face is too close; likely only head/face visible"], warnings, scores)
        if face_height_ratio < cfg.min_face_height_ratio:
            return _result("manual_verification", ["Face is very small in the image"], warnings, scores)

    if space_below_face_ratio is not None:
        if space_below_face_ratio < cfg.min_space_below_face_reject:
            return _result("rejected", ["Not enough area below face for upper body"], warnings, scores)
        if space_below_face_ratio < cfg.min_space_below_face_manual:
            warnings.append("Limited area below face; crop may be head-and-shoulders only")

    body_status = body.get("body_status")
    if body_status == "more_than_head_shoulders":
        reasons.append("One clear face detected")
        reasons.append("Pose landmarks show more than head and shoulders")
        
        if ai_flag == "manual_verification":
            reasons.append("AI-generated image suspicion is medium or high")
            return _result("manual_verification", reasons, warnings, scores)
            
        if warnings:
            return _result("manual_verification", reasons, warnings, scores)
        return _result("acceptable", reasons, warnings, scores)

    if body_status == "head_shoulders_only":
        return _result("rejected", ["Only head and shoulders appear visible"], warnings, scores)

    if body_status in {"pose_not_found", "manual_body_partial"}:
        return _result("manual_verification", ["Body visibility could not be confirmed confidently"], warnings, scores)

    if body_status == "face_or_head_only":
        return _result("rejected", ["Body landmarks do not show more than head/shoulders"], warnings, scores)

    return _result("manual_verification", ["Unable to verify body visibility"], warnings, scores)


def _result(flag: str, reasons: list[str], warnings: list[str], scores: dict) -> dict:
    return {
        "flag": flag,
        "reasons": reasons,
        "warnings": warnings,
        "scores": scores,
    }

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import distance2bbox, distance2kps, nms


class SCRFDFaceDetector:
    """SCRFD ONNX face detector using ONNX Runtime.

    Expected model: SCRFD 2.5G KPS/BNKPS ONNX, for example:
    models/scrfd_2.5g_bnkps.onnx
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        det_threshold: float = 0.50,
        nms_threshold: float = 0.40,
        providers: list[str] | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SCRFD ONNX model not found: {model_path}. "
                "Put scrfd_2.5g_bnkps.onnx inside the models/ folder."
            )

        self.model_path = model_path
        self.input_size = input_size  # (width, height)
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image_bgr: np.ndarray) -> dict[str, Any]:
        original_h, original_w = image_bgr.shape[:2]
        input_w, input_h = self.input_size

        resized_img, det_scale = self._resize_and_pad(image_bgr, input_w, input_h)

        blob = cv2.dnn.blobFromImage(
            resized_img,
            scalefactor=1.0 / 128.0,
            size=(input_w, input_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )

        outputs = self.session.run(None, {self.input_name: blob})
        scores_list, bboxes_list, kpss_list = self._decode_outputs(outputs, input_w, input_h)

        if len(scores_list) == 0:
            return {"face_count": 0, "faces": []}

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / det_scale
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_h - 1)

        if len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
            kpss[:, 0::2] = np.clip(kpss[:, 0::2], 0, original_w - 1)
            kpss[:, 1::2] = np.clip(kpss[:, 1::2], 0, original_h - 1)
        else:
            kpss = None

        pre_det = np.hstack((bboxes, scores.reshape(-1, 1))).astype(np.float32, copy=False)
        keep = nms(pre_det, self.nms_threshold)
        dets = pre_det[keep]
        kpss = kpss[keep] if kpss is not None and len(keep) > 0 else None

        # Sort largest/highest confidence first. This helps when a poster/background face is detected.
        if len(dets) > 0:
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            order = np.lexsort((-areas, -dets[:, 4]))
            dets = dets[order]
            if kpss is not None:
                kpss = kpss[order]

        faces = []
        for idx, det in enumerate(dets):
            x1, y1, x2, y2, score = det.tolist()
            face = {
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(float(score), 4),
            }
            if kpss is not None:
                pts = kpss[idx].reshape(-1, 2).tolist()
                face["landmarks"] = [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
            faces.append(face)

        return {"face_count": len(faces), "faces": faces}

    def _resize_and_pad(self, image_bgr: np.ndarray, input_w: int, input_h: int) -> tuple[np.ndarray, float]:
        h, w = image_bgr.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h))
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, scale

    def _decode_outputs(self, outputs: list[np.ndarray], input_w: int, input_h: int):
        # SCRFD usually returns 9 outputs for KPS models:
        # scores for strides 8/16/32, bbox distances, keypoint distances.
        output_count = len(outputs)
        if output_count % 3 == 0:
            feature_count = output_count // 3
        elif output_count % 2 == 0:
            feature_count = output_count // 2
        else:
            raise RuntimeError(f"Unexpected SCRFD output count: {output_count}")

        strides = self.strides[:feature_count]
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(strides):
            scores = outputs[idx]
            bbox_preds = outputs[idx + feature_count] * stride
            kps_preds = outputs[idx + feature_count * 2] * stride if output_count >= feature_count * 3 else None

            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            if kps_preds is not None:
                kps_preds = kps_preds.reshape(-1, 10)

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            if key not in self.center_cache:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
                self.center_cache[key] = anchor_centers

            anchor_centers = self.center_cache[key]
            positive_indices = np.where(scores >= self.det_threshold)[0]
            if len(positive_indices) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            selected_scores = scores[positive_indices]
            selected_bboxes = bboxes[positive_indices]

            scores_list.append(selected_scores.reshape(-1, 1))
            bboxes_list.append(selected_bboxes)

            if kps_preds is not None:
                kpss = distance2kps(anchor_centers, kps_preds)
                selected_kpss = kpss[positive_indices]
                kpss_list.append(selected_kpss)

        return scores_list, bboxes_list, kpss_list

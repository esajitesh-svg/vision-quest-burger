"""Per-frame local object detection with OpenCV DNN and the OpenCV Zoo YOLOX model.

The model is downloaded once per server cache and inference runs on the app host
for every video frame delivered to the WebRTC callback. No frame is sent to a
remote object-detection API. The separate scene-explanation VLM remains remote.
Model source: OpenCV Zoo, object_detection_yolox, Apache-2.0 model-directory license.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

REALTIME_DETECTOR_LABEL = "OpenCV Zoo YOLOX-S INT8 · OpenCV DNN CPU · COCO-80"
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/"
    "object_detection_yolox/object_detection_yolox_2022nov_int8.onnx?download=true"
)
MODEL_FILENAME = "visionquest_object_detection_yolox_int8.onnx"
MODEL_MIN_BYTES = 7 * 1024 * 1024
MODEL_MAX_BYTES = 18 * 1024 * 1024
MODEL_INPUT_SIZE = 640
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
_STRIDES = (8, 16, 32)
_DOWNLOAD_LOCK = threading.Lock()


def _model_path() -> Path:
    """Download and atomically cache the compact OpenCV Zoo ONNX model."""
    configured_dir = os.getenv("VISIONQUEST_MODEL_CACHE", "").strip()
    cache_dir = Path(configured_dir) if configured_dir else Path(tempfile.gettempdir()) / "visionquest-models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / MODEL_FILENAME

    with _DOWNLOAD_LOCK:
        if target.is_file() and MODEL_MIN_BYTES <= target.stat().st_size <= MODEL_MAX_BYTES:
            return target

        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=(10, 90), allow_redirects=True) as response:
                response.raise_for_status()
                written = 0
                with open(temporary, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > MODEL_MAX_BYTES:
                            raise RuntimeError("The downloaded object detector exceeded its expected size limit.")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if not MODEL_MIN_BYTES <= written <= MODEL_MAX_BYTES:
                raise RuntimeError(
                    "The object-detector download was incomplete or unexpected in size. "
                    "Check the hosting network and retry the app."
                )
            os.replace(temporary, target)
        except requests.RequestException as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                "Could not download the local YOLOX model from OpenCV Zoo. "
                "Check whether the cloud host can reach Hugging Face, then reboot."
            ) from exc
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return target


@dataclass(frozen=True)
class RealtimeDetectionResult:
    detections: list[dict[str, Any]]
    elapsed_seconds: float
    image_size: tuple[int, int]
    fps: float


class RealtimeYOLOXDetector:
    """Thread-safe YOLOX object detector using only the installed OpenCV DNN runtime."""

    def __init__(self, model_path: str | Path | None = None, net=None):
        self.input_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        self.net = net if net is not None else cv2.dnn.readNet(str(model_path or _model_path()))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._infer_lock = threading.Lock()
        self._grids, self._expanded_strides = self._create_grids()

    @staticmethod
    def _create_grids() -> tuple[np.ndarray, np.ndarray]:
        grids = []
        expanded_strides = []
        for stride in _STRIDES:
            grid_size = MODEL_INPUT_SIZE // stride
            xv, yv = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride, dtype=np.float32))
        return (
            np.concatenate(grids, axis=1).astype(np.float32),
            np.concatenate(expanded_strides, axis=1).astype(np.float32),
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Match OpenCV Zoo YOLOX input: RGB image, 640-square top-left letterbox, NCHW float."""
        height, width = frame_bgr.shape[:2]
        ratio = min(self.input_size[0] / height, self.input_size[1] / width)
        resized_w = max(1, int(width * ratio))
        resized_h = max(1, int(height * ratio))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114.0, dtype=np.float32)
        canvas[:resized_h, :resized_w] = resized.astype(np.float32)
        nchw = np.ascontiguousarray(np.transpose(canvas, (2, 0, 1))[None, :, :, :], dtype=np.float32)
        return nchw, ratio

    def _decode(
        self,
        output: np.ndarray,
        ratio: float,
        image_width: int,
        image_height: int,
        confidence_threshold: float,
        nms_threshold: float = 0.45,
        max_detections: int = 30,
    ) -> list[dict[str, Any]]:
        """Decode YOLOX raw output, apply per-class non-maximum suppression, and map boxes back."""
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 5 + len(COCO_CLASSES):
            raise RuntimeError(f"The local YOLOX model returned an unexpected output shape: {predictions.shape}.")

        predictions = predictions.astype(np.float32, copy=True)
        count = min(predictions.shape[0], self._grids.shape[1], self._expanded_strides.shape[1])
        predictions = predictions[:count]
        predictions[:, :2] = (predictions[:, :2] + self._grids[0, :count]) * self._expanded_strides[0, :count]
        predictions[:, 2:4] = np.exp(np.clip(predictions[:, 2:4], -10.0, 10.0)) * self._expanded_strides[0, :count, :]

        boxes = predictions[:, :4]
        corners = np.empty_like(boxes)
        corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        class_scores = predictions[:, 4:5] * predictions[:, 5:5 + len(COCO_CLASSES)]
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)
        valid = np.isfinite(scores) & np.all(np.isfinite(corners), axis=1) & (scores >= confidence_threshold)

        selected: list[dict[str, Any]] = []
        for class_id in np.unique(class_ids[valid]):
            class_indices = np.flatnonzero(valid & (class_ids == class_id))
            class_boxes_xywh = []
            class_scores_list = []
            for idx in class_indices:
                x1, y1, x2, y2 = corners[idx]
                class_boxes_xywh.append((
                    float(x1),
                    float(y1),
                    max(0.0, float(x2 - x1)),
                    max(0.0, float(y2 - y1)),
                ))
                class_scores_list.append(float(scores[idx]))
            keep = cv2.dnn.NMSBoxes(
                class_boxes_xywh, class_scores_list,
                float(confidence_threshold), float(nms_threshold)
            )
            keep_indices = np.asarray(keep).reshape(-1).tolist() if len(keep) else []
            for keep_idx in keep_indices:
                idx = int(class_indices[int(keep_idx)])
                x1, y1, x2, y2 = corners[idx] / float(ratio)
                x1 = max(0, min(image_width - 1, int(round(float(x1)))))
                y1 = max(0, min(image_height - 1, int(round(float(y1)))))
                x2 = max(0, min(image_width - 1, int(round(float(x2)))))
                y2 = max(0, min(image_height - 1, int(round(float(y2)))))
                if x2 <= x1 or y2 <= y1:
                    continue
                selected.append({
                    "label": COCO_CLASSES[int(class_id)],
                    "score": max(0.0, min(1.0, float(scores[idx]))),
                    "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                })
        selected.sort(key=lambda item: item["score"], reverse=True)
        return selected[:max(1, min(50, int(max_detections)))]

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.35,
        max_detections: int = 30,
    ) -> RealtimeDetectionResult:
        """Run local inference on a single BGR frame and return detections for that same frame."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            raise ValueError("Object detection received an empty camera frame.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Object detection expects a three-channel BGR frame.")
        if not 0.05 <= float(confidence_threshold) <= 0.95:
            raise ValueError("Detection threshold must be between 5% and 95%.")

        image_height, image_width = frame_bgr.shape[:2]
        blob, ratio = self._preprocess(frame_bgr)
        started = time.perf_counter()
        try:
            with self._infer_lock:
                self.net.setInput(blob)
                outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV DNN could not run the YOLOX model on this host CPU. "
                "The camera can still be used without object labels."
            ) from exc
        elapsed = max(0.001, time.perf_counter() - started)
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise RuntimeError("OpenCV DNN returned no output from the YOLOX model.")
        detections = self._decode(
            outputs[0], ratio, image_width, image_height,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        )
        total_elapsed = max(0.001, time.perf_counter() - started)
        return RealtimeDetectionResult(
            detections=detections,
            elapsed_seconds=round(total_elapsed, 4),
            image_size=(image_width, image_height),
            fps=round(1.0 / total_elapsed, 2),
        )

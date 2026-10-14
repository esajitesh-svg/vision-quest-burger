"""Per-session OpenCV live-stream state and silent MP4 recording helpers.

The browser captures and streams video over WebRTC. OpenCV runs on the cloud host.
Video clips are stored temporarily while recording, exposed for download after stop,
and removed from the temporary path after finalization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np

MAX_RECORD_SECONDS = 60
MAX_VIDEO_BYTES = 25 * 1024 * 1024
RECORD_FPS = 8.0
MAX_FRAME_EDGE = 1920


def process_live_frame(
    frame_bgr: np.ndarray,
    max_side: int = MAX_FRAME_EDGE,
    enhance: bool = False,
    encode_jpeg: bool = True,
):
    """Resize and inspect a BGR video frame; optionally encode it for later analysis."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Received an empty camera frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Camera frame must have three color channels.")
    max_side = max(160, min(int(max_side), MAX_FRAME_EDGE))
    h0, w0 = frame_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(w0, h0)))
    if scale < 1.0:
        frame = cv2.resize(
            frame_bgr,
            (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        frame = frame_bgr.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Focus/brightness metrics need no full-resolution float64 Laplacian on every frame.
    # Estimate detail on a <=640 px grayscale sample to reduce per-frame CPU/memory cost.
    metric_scale = min(1.0, 640.0 / float(max(gray.shape[1], gray.shape[0])))
    metric_gray = (
        cv2.resize(
            gray,
            (max(1, int(round(gray.shape[1] * metric_scale))), max(1, int(round(gray.shape[0] * metric_scale)))),
            interpolation=cv2.INTER_AREA,
        )
        if metric_scale < 1.0 else gray
    )
    brightness = float(cv2.mean(metric_gray)[0])
    sharpness = float(cv2.Laplacian(metric_gray, cv2.CV_32F).var())

    if enhance:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        frame = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    encoded_bytes = b""
    if encode_jpeg:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("OpenCV could not encode this camera frame.")
        encoded_bytes = encoded.tobytes()
    height, width = frame.shape[:2]
    metrics = {
        "width": int(width),
        "height": int(height),
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "brightness_label": "dark" if brightness < 55 else ("bright" if brightness > 205 else "normal"),
        "sharpness_label": "possibly soft/blurry" if sharpness < 45 else "reasonable edge detail",
        "jpeg_bytes": len(encoded_bytes),
        "jpeg_quality": 90,
    }
    return frame, encoded_bytes, metrics


def render_cv_view(
    frame_bgr: np.ndarray,
    mode: str = "COLOR + HUD",
    canny_low: int = 55,
    canny_high: int = 140,
    frame_count: int = 0,
) -> np.ndarray:
    """Render a live OpenCV diagnostic view without changing the AI input frame."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Cannot render an empty OpenCV frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("OpenCV display input must have three color channels.")

    source = frame_bgr.copy()
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    low = max(0, min(254, int(canny_low)))
    high = max(low + 1, min(255, int(canny_high)))

    if mode == "GRAYSCALE":
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "CANNY EDGE MAP":
        edges = cv2.Canny(gray, low, high)
        display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == "CONTOUR MAP":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = source.copy()
        visible = [contour for contour in contours if cv2.contourArea(contour) >= 120]
        cv2.drawContours(display, visible, -1, (70, 255, 115), 2)
        for contour in visible[:30]:
            x, y, width, height = cv2.boundingRect(contour)
            if width > 24 and height > 20:
                cv2.rectangle(display, (x, y), (x + width, y + height), (255, 210, 60), 1)
    elif mode == "ADAPTIVE THRESHOLD":
        mask = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif mode == "HSV SATURATION":
        saturation = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[:, :, 1]
        display = cv2.applyColorMap(saturation, cv2.COLORMAP_TURBO)
    else:
        # COLOR + HUD is the default and preserves the normal color image.
        display = source.copy()
        height, width = display.shape[:2]
        cx, cy = width // 2, height // 2
        hud_color = (255, 235, 70)
        cv2.line(display, (cx - 22, cy), (cx - 7, cy), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx + 7, cy), (cx + 22, cy), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy - 22), (cx, cy - 7), hud_color, 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy + 7), (cx, cy + 22), hud_color, 1, cv2.LINE_AA)
        cv2.circle(display, (cx, cy), 4, hud_color, 1, cv2.LINE_AA)
        arm = max(12, min(30, width // 18, height // 12))
        margin = 10
        for x, y, sx, sy in [
            (margin, margin, 1, 1),
            (width - margin, margin, -1, 1),
            (margin, height - margin, 1, -1),
            (width - margin, height - margin, -1, -1),
        ]:
            cv2.line(display, (x, y), (x + sx * arm, y), hud_color, 2, cv2.LINE_AA)
            cv2.line(display, (x, y), (x, y + sy * arm), hud_color, 2, cv2.LINE_AA)

    height, width = display.shape[:2]
    # Scanner rails and telemetry make the feed a working vision canvas, not a webpage card.
    cv2.rectangle(display, (0, 0), (width, 32), (4, 14, 20), thickness=-1)
    cv2.rectangle(display, (0, height - 25), (width, height), (4, 14, 20), thickness=-1)
    label = f"VISIONQUEST // {mode}"
    cv2.putText(display, label[:64], (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (90, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(
        display,
        f"FRAME {max(0, int(frame_count)):06d}  |  {width}x{height}  |  OPENCV PIPELINE",
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (165, 222, 230),
        1,
        cv2.LINE_AA,
    )
    return display


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    age_seconds: float | None = None,
    max_age_seconds: float = 9.0,
) -> np.ndarray:
    """Draw labelled confidence boxes over a live frame while results are fresh."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Cannot draw detections on an empty frame.")
    display = frame_bgr.copy()
    if age_seconds is not None and age_seconds > max_age_seconds:
        return display

    height, width = display.shape[:2]
    color = (90, 245, 160)  # BGR neon green
    for item in detections[:30]:
        box = item.get("box", {})
        try:
            x1 = max(0, min(width - 1, int(round(float(box["xmin"])))))
            y1 = max(0, min(height - 1, int(round(float(box["ymin"])))))
            x2 = max(0, min(width - 1, int(round(float(box["xmax"])))))
            y2 = max(0, min(height - 1, int(round(float(box["ymax"])))))
            score = max(0.0, min(1.0, float(item.get("score", 0.0))))
            label = str(item.get("label", "object"))[:32]
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = f"{label.upper()}  {score:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
        )
        tag_y = max(33, y1 - text_height - baseline - 7)
        tag_x2 = min(width - 1, x1 + text_width + 10)
        cv2.rectangle(display, (x1, tag_y), (tag_x2, tag_y + text_height + baseline + 6), color, -1)
        cv2.putText(
            display, text, (x1 + 4, tag_y + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (3, 14, 18), 1, cv2.LINE_AA
        )
    return display


class LiveMonitorState:
    """Thread-safe state owned by one Streamlit browser session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.latest_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_analysis_jpeg: bytes | None = None
        self.latest_analysis_frame_bgr: np.ndarray | None = None
        self.latest_metrics: dict[str, Any] = {}
        self.frame_count = 0
        self.last_frame_time = 0.0

        self.recording = False
        self.recording_started = 0.0
        self.recording_limit = 30
        self.video_writer = None
        self.video_path: str | None = None
        self.video_size: tuple[int, int] | None = None
        self.video_frame_count = 0
        self.completed_video: bytes | None = None
        self.completed_filename = "visionquest-live-recording.mp4"
        self.recording_notice = ""

        self.analysis_busy = False
        self.last_analysis_started: float | None = None
        self.latest_observation: dict[str, Any] | None = None
        self.latest_analysis_error = ""

        # Object detection runs independently from the scene-description model.
        self.detection_busy = False
        self.last_detection_started: float | None = None
        self.latest_detections: list[dict[str, Any]] = []
        self.latest_detection_error = ""
        self.latest_detection_completed: float | None = None
        self.latest_detection_elapsed = 0.0
        self.latest_detection_model = ""

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        jpeg_bytes: bytes,
        metrics: dict[str, Any],
        analysis_jpeg: bytes | None = None,
        analysis_frame_bgr: np.ndarray | None = None,
    ) -> None:
        """Called from WebRTC's video callback thread. Do not call Streamlit here."""
        now = time.monotonic()
        with self.lock:
            self.latest_frame = frame_bgr.copy()
            self.latest_jpeg = bytes(jpeg_bytes)
            # Hold the clean BGR frame for AI analysis and encode it only when a
            # scene-analysis request is actually reserved, not on every video frame.
            if analysis_frame_bgr is not None:
                self.latest_analysis_frame_bgr = analysis_frame_bgr
                self.latest_analysis_jpeg = None
            else:
                self.latest_analysis_frame_bgr = None
                self.latest_analysis_jpeg = bytes(analysis_jpeg if analysis_jpeg is not None else jpeg_bytes)
            self.latest_metrics = dict(metrics)
            self.frame_count += 1
            self.last_frame_time = now

            if not self.recording or self.video_writer is None:
                return

            elapsed = now - self.recording_started
            if elapsed >= self.recording_limit:
                self._finalize_recording_locked(f"Recording stopped at the {self.recording_limit}-second limit.")
                return

            target_w, target_h = self.video_size or (frame_bgr.shape[1], frame_bgr.shape[0])
            record_frame = frame_bgr
            if record_frame.shape[1] != target_w or record_frame.shape[0] != target_h:
                record_frame = cv2.resize(record_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self.video_writer.write(record_frame)
            self.video_frame_count += 1

            if self.video_frame_count % 8 == 0 and self.video_path:
                try:
                    if os.path.getsize(self.video_path) >= MAX_VIDEO_BYTES:
                        self._finalize_recording_locked("Recording stopped to keep the clip below the 25 MB limit.")
                except OSError:
                    self._finalize_recording_locked("Recording stopped because its temporary file became unavailable.")

    def snapshot(self):
        with self.lock:
            return (
                bytes(self.latest_jpeg) if self.latest_jpeg else None,
                dict(self.latest_metrics),
                self.frame_count,
                self.last_frame_time,
            )

    def start_recording(self, seconds: int = 30) -> None:
        with self.lock:
            if self.recording:
                raise RuntimeError("A recording is already in progress.")
            if self.latest_frame is None:
                raise RuntimeError("Start the webcam stream and wait for a camera frame before recording.")
            limit = min(MAX_RECORD_SECONDS, max(5, int(seconds)))
            h, w = self.latest_frame.shape[:2]
            w = max(2, w - (w % 2))
            h = max(2, h - (h % 2))
            fd, path = tempfile.mkstemp(prefix="visionquest_", suffix=".mp4")
            os.close(fd)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, RECORD_FPS, (w, h))
            if not writer.isOpened():
                writer.release()
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise RuntimeError("This cloud host could not initialize an MP4 encoder.")
            self.video_writer = writer
            self.video_path = path
            self.video_size = (w, h)
            self.video_frame_count = 0
            self.recording_limit = limit
            self.recording_started = time.monotonic()
            self.recording = True
            self.completed_video = None
            self.recording_notice = f"Recording started; it will stop automatically after {limit} seconds."

    def stop_recording(self, notice: str = "Recording stopped by user.") -> None:
        with self.lock:
            if self.recording or self.video_writer is not None:
                self._finalize_recording_locked(notice)

    def _finalize_recording_locked(self, notice: str) -> None:
        writer = self.video_writer
        self.video_writer = None
        self.recording = False
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        path = self.video_path
        self.video_path = None
        self.recording_notice = notice
        if path and os.path.isfile(path):
            try:
                size = os.path.getsize(path)
                if 0 < size <= MAX_VIDEO_BYTES:
                    with open(path, "rb") as handle:
                        video = handle.read()
                    if video:
                        self.completed_video = video
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
                        self.completed_filename = f"visionquest_recording_{stamp}.mp4"
                elif size > MAX_VIDEO_BYTES:
                    self.completed_video = None
                    self.recording_notice = "Recording exceeded the 25 MB safety limit; it was not kept for download."
                else:
                    self.completed_video = None
                    self.recording_notice = "Recording contained no usable frames. Start again after the preview shows live frames."
            except OSError:
                self.completed_video = None
                self.recording_notice = "Recording could not be finalized for download."
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        self.video_size = None

    def take_analysis_sample(
        self,
        interval_seconds: int,
        force: bool = False,
        max_side: int = 768,
    ) -> bytes | None:
        """Reserve the newest frame and encode it at the requested detail only once."""
        now = time.monotonic()
        with self.lock:
            if self.analysis_busy or (
                not self.latest_analysis_jpeg and self.latest_analysis_frame_bgr is None
            ):
                return None
            interval = min(60, max(1, int(interval_seconds)))
            if not force and self.last_analysis_started is not None and now - self.last_analysis_started < interval:
                return None
            self.analysis_busy = True
            self.last_analysis_started = now
            if self.latest_analysis_jpeg:
                return bytes(self.latest_analysis_jpeg)
            frame = self.latest_analysis_frame_bgr.copy()

        max_side = max(64, min(int(max_side), MAX_FRAME_EDGE))
        height, width = frame.shape[:2]
        scale = min(1.0, max_side / float(max(width, height)))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            self.finish_analysis(error="OpenCV could not encode a frame for scene analysis.")
            return None
        return encoded.tobytes()

    def finish_analysis(self, observation: dict[str, Any] | None = None, error: str = "") -> None:
        with self.lock:
            self.analysis_busy = False
            # The interval is measured between request starts. If inference
            # itself takes longer than the interval, the next request may start
            # immediately after completion; analysis requests never overlap.
            self.latest_analysis_error = error[:500]
            if observation:
                self.latest_observation = dict(observation)

    def reserve_live_detection(self, interval_seconds: float = 0.6) -> bool:
        """Reserve one asynchronous local-inference job without blocking the camera callback."""
        now = time.monotonic()
        with self.lock:
            if self.detection_busy:
                return False
            interval = max(0.2, min(5.0, float(interval_seconds)))
            if (
                self.last_detection_started is not None
                and now - self.last_detection_started < interval
            ):
                return False
            self.detection_busy = True
            self.last_detection_started = now
            self.latest_detection_error = ""
            return True

    def take_detection_sample(self, interval_seconds: int = 3, force: bool = False) -> bytes | None:
        """Reserve the newest color frame for one remote object-detection request."""
        now = time.monotonic()
        with self.lock:
            if self.detection_busy or not self.latest_analysis_jpeg:
                return None
            interval = min(60, max(1, int(interval_seconds)))
            if not force and self.last_detection_started is not None and now - self.last_detection_started < interval:
                return None
            self.detection_busy = True
            self.last_detection_started = now
            return bytes(self.latest_analysis_jpeg)

    def finish_detection(
        self,
        detections: list[dict[str, Any]] | None = None,
        model: str = "",
        elapsed_seconds: float = 0.0,
        error: str = "",
    ) -> None:
        """Publish one detector result from a worker thread."""
        with self.lock:
            self.detection_busy = False
            self.latest_detection_error = error[:500]
            if detections is not None and not error:
                self.latest_detections = [dict(item) for item in detections]
                self.latest_detection_model = str(model)
                self.latest_detection_elapsed = max(0.0, float(elapsed_seconds))
                self.latest_detection_completed = time.monotonic()

    def detection_snapshot(self) -> dict[str, Any]:
        """Get a thread-safe copy of the latest object-detection result."""
        with self.lock:
            age = None
            if self.latest_detection_completed is not None:
                age = max(0.0, time.monotonic() - self.latest_detection_completed)
            return {
                "detections": [dict(item) for item in self.latest_detections],
                "busy": self.detection_busy,
                "error": self.latest_detection_error,
                "age_seconds": age,
                "elapsed_seconds": self.latest_detection_elapsed,
                "model": self.latest_detection_model,
            }

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            duration = (time.monotonic() - self.recording_started) if self.recording else 0.0
            return {
                "recording": self.recording,
                "recording_duration": round(duration, 1),
                "recording_limit": self.recording_limit,
                "recording_notice": self.recording_notice,
                "video_frame_count": self.video_frame_count,
                "completed_video_size": len(self.completed_video) if self.completed_video else 0,
                "completed_filename": self.completed_filename,
                "analysis_busy": self.analysis_busy,
                "latest_observation": dict(self.latest_observation) if self.latest_observation else None,
                "latest_analysis_error": self.latest_analysis_error,
                "detection_busy": self.detection_busy,
                "latest_detections": [dict(item) for item in self.latest_detections],
                "latest_detection_error": self.latest_detection_error,
                "latest_detection_elapsed": self.latest_detection_elapsed,
                "latest_detection_model": self.latest_detection_model,
                "frame_count": self.frame_count,
                "latest_metrics": dict(self.latest_metrics),
            }

    def get_completed_video(self) -> tuple[bytes | None, str]:
        with self.lock:
            return self.completed_video, self.completed_filename

    def clear_completed_video(self) -> None:
        with self.lock:
            self.completed_video = None

    def close(self) -> None:
        self.stop_recording("Recording closed because the session ended.")

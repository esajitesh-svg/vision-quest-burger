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
MAX_FRAME_EDGE = 640


def process_live_frame(frame_bgr: np.ndarray, max_side: int = MAX_FRAME_EDGE, enhance: bool = False):
    """Resize and inspect a single BGR video frame; return frame, JPEG bytes, metrics."""
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
    brightness = float(cv2.mean(gray)[0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if enhance:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        frame = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("OpenCV could not encode this camera frame.")
    height, width = frame.shape[:2]
    metrics = {
        "width": int(width),
        "height": int(height),
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "brightness_label": "dark" if brightness < 55 else ("bright" if brightness > 205 else "normal"),
        "sharpness_label": "possibly soft/blurry" if sharpness < 45 else "reasonable edge detail",
        "jpeg_bytes": int(encoded.nbytes),
    }
    return frame, encoded.tobytes(), metrics


class LiveMonitorState:
    """Thread-safe state owned by one Streamlit browser session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.latest_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
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

    def update_frame(self, frame_bgr: np.ndarray, jpeg_bytes: bytes, metrics: dict[str, Any]) -> None:
        """Called from WebRTC's video callback thread. Do not call Streamlit here."""
        now = time.monotonic()
        with self.lock:
            self.latest_frame = frame_bgr.copy()
            self.latest_jpeg = bytes(jpeg_bytes)
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

    def take_analysis_sample(self, interval_seconds: int, force: bool = False) -> bytes | None:
        """Reserve newest frame for one analysis; force bypasses sampling interval."""
        now = time.monotonic()
        with self.lock:
            if self.analysis_busy or not self.latest_jpeg:
                return None
            interval = min(60, max(10, int(interval_seconds)))
            if not force and self.last_analysis_started is not None and now - self.last_analysis_started < interval:
                return None
            self.analysis_busy = True
            self.last_analysis_started = now
            return bytes(self.latest_jpeg)

    def finish_analysis(self, observation: dict[str, Any] | None = None, error: str = "") -> None:
        with self.lock:
            self.analysis_busy = False
            # Measure the cadence from completion, not start, so a slow model
            # cannot trigger back-to-back inferences when a fragment reruns.
            self.last_analysis_started = time.monotonic()
            self.latest_analysis_error = error[:500]
            if observation:
                self.latest_observation = dict(observation)

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

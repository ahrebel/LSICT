"""People and face detection.

People: YOLOv8n (Ultralytics), class 0 = person.
Face:   MediaPipe FaceDetection (preferred) or OpenCV Haar cascade (fallback).

Both stages cache results in the SQLite cache, so re-runs only process files
that have changed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from lsict.cache import Cache
from lsict.core import (
    safe_open_pil,
    yolo_device_arg,
)

logger = logging.getLogger(__name__)


# ---------- YOLO loading ----------

def load_yolo(device: str):
    """Load YOLOv8n. Downloads weights on first run."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    # ultralytics defers device assignment until inference; we pass device per-call
    return model


# ---------- Face backends ----------

class HaarFaceDetector:
    """OpenCV Haar cascade face detector. Cheap, decent for frontal faces."""

    def __init__(self, min_size: tuple[int, int] = (40, 40)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.min_size = min_size

    def has_face(self, path: Path) -> bool:
        try:
            # cv2.imdecode handles non-ASCII paths on Windows (cv2.imread can't).
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return False
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            faces = self.cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=4, minSize=self.min_size
            )
            return len(faces) > 0
        except Exception as e:
            logger.debug("Haar face detect failed on %s: %s", path, e)
            return False


class MediaPipeFaceDetector:
    """MediaPipe FaceDetection — handles profiles & angled faces much better than Haar."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe not installed. Install with: pip install mediapipe"
            ) from e
        self.mp = mp
        # model_selection=1 is better for general use; 0 is for close-up selfies.
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_confidence
        )

    def has_face(self, path: Path) -> bool:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return False
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return False
            # MediaPipe expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)
            return bool(results.detections)
        except Exception as e:
            logger.debug("MediaPipe face detect failed on %s: %s", path, e)
            return False


def make_face_detector(backend: str):
    """Build a face detector. backend: 'mediapipe', 'haar', 'auto', or 'off'."""
    if backend == "off":
        return None
    if backend in ("mediapipe", "auto"):
        try:
            return MediaPipeFaceDetector()
        except RuntimeError as e:
            if backend == "mediapipe":
                raise
            logger.info("MediaPipe unavailable (%s); falling back to Haar cascade", e)
    return HaarFaceDetector()


# ---------- Orchestration ----------

def run_detection(
    files: list[Path],
    cache: Cache,
    device: str,
    face_backend: str = "auto",
    yolo_batch: int = 32,
    yolo_conf: float = 0.25,
    skip_yolo: bool = False,
) -> dict[Path, dict]:
    """Run people+face detection over all files. Cached results are reused.

    Returns: {Path: {"has_person": bool, "has_face": bool}}
    """
    results: dict[Path, dict] = {}

    # 1) Partition into cached vs needs-detection
    need_detect: list[Path] = []
    for p in files:
        cached = cache.get_detection(p)
        if cached is not None:
            results[p] = {"has_person": cached[0], "has_face": cached[1]}
        else:
            need_detect.append(p)

    if not need_detect:
        logger.info("Detection: all %d files served from cache.", len(files))
        return results

    logger.info(
        "Detection: %d cached, %d need processing (%.1f%% cache hit)",
        len(files) - len(need_detect),
        len(need_detect),
        100.0 * (len(files) - len(need_detect)) / max(1, len(files)),
    )

    # 2) YOLO person detection
    person_flags: dict[Path, bool] = {p: False for p in need_detect}

    if not skip_yolo:
        yolo = load_yolo(device)
        ydev = yolo_device_arg(device)
        try:
            pbar = tqdm(total=len(need_detect), desc="YOLO person", unit="img")
            for i in range(0, len(need_detect), yolo_batch):
                chunk = need_detect[i:i + yolo_batch]
                try:
                    res_list = yolo(
                        [str(p) for p in chunk],
                        conf=yolo_conf,
                        classes=[0],   # 0 = person
                        device=ydev,
                        verbose=False,
                    )
                except Exception as e:
                    logger.warning("YOLO batch failed: %s", e)
                    res_list = [None] * len(chunk)

                for p, res in zip(chunk, res_list):
                    has_person = False
                    try:
                        if res is not None and getattr(res, "boxes", None) is not None:
                            has_person = len(res.boxes) > 0
                    except Exception:
                        has_person = False
                    person_flags[p] = has_person
                pbar.update(len(chunk))
            pbar.close()
        finally:
            # Free model memory between stages
            del yolo

    # 3) Face detection on files NOT already flagged by YOLO
    face_flags: dict[Path, bool] = {p: False for p in need_detect}
    face_detector = make_face_detector(face_backend)
    if face_detector is not None:
        unflagged = [p for p in need_detect if not person_flags.get(p, False)]
        if unflagged:
            backend_name = type(face_detector).__name__.replace("FaceDetector", "")
            for p in tqdm(unflagged, desc=f"Face detect ({backend_name})", unit="img"):
                face_flags[p] = face_detector.has_face(p)

    # 4) Persist to cache
    with cache.transaction():
        for p in need_detect:
            cache.set_detection(p, person_flags[p], face_flags[p])
            results[p] = {
                "has_person": person_flags[p],
                "has_face": face_flags[p],
            }

    n_person = sum(1 for v in results.values() if v["has_person"])
    n_face = sum(1 for v in results.values()
                 if not v["has_person"] and v["has_face"])
    logger.info(
        "Detection complete: %d people, %d face-only, %d clean (of %d total)",
        n_person, n_face, len(files) - n_person - n_face, len(files),
    )
    return results

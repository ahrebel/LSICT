"""People and face detection.

People: YOLOv8n (Ultralytics), class 0 = person.
Face:   MediaPipe FaceDetection, OpenCV YuNet (FaceDetectorYN), or OpenCV
        Haar cascade. 'auto' tries them in that order. Note that Haar's
        CascadeClassifier API was removed in OpenCV 5 — YuNet works on both
        OpenCV 4 (>= 4.5.4) and 5.

Results are cached incrementally (per YOLO batch / face chunk), so an
interrupted or crashed run resumes where it left off.
"""
from __future__ import annotations

import logging
import os
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
    """OpenCV Haar cascade face detector. Cheap, decent for frontal faces.

    Only available on OpenCV 4 — the CascadeClassifier API was removed in
    OpenCV 5 (use YuNet there).
    """

    def __init__(self, min_size: tuple[int, int] = (40, 40)):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no CascadeClassifier (removed in "
                "OpenCV 5) — use the 'yunet' or 'mediapipe' face backend instead"
            )
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


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _yunet_model_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    d = cache_root / "lsict"
    d.mkdir(parents=True, exist_ok=True)
    return d / "face_detection_yunet_2023mar.onnx"


class YuNetFaceDetector:
    """OpenCV YuNet face detector (cv2.FaceDetectorYN, OpenCV >= 4.5.4).

    Better than Haar on angled/partial faces, and works on OpenCV 5 (which
    removed CascadeClassifier). The ONNX model (~230 KB) downloads on first
    use into the lsict cache directory.
    """

    def __init__(self, score_threshold: float = 0.6, max_side: int = 1024):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                f"cv2.FaceDetectorYN unavailable (OpenCV {cv2.__version__}; "
                "need >= 4.5.4)"
            )
        model_path = _yunet_model_path()
        if not model_path.exists():
            logger.info("Downloading YuNet face model (~230 KB)...")
            import urllib.request
            req = urllib.request.Request(
                YUNET_URL, headers={"User-Agent": "lsict"})
            tmp = model_path.with_suffix(".onnx.tmp")
            try:
                with urllib.request.urlopen(req, timeout=60) as r, \
                        open(tmp, "wb") as f:
                    f.write(r.read())
                os.replace(tmp, model_path)
            except Exception as e:
                if tmp.exists():
                    tmp.unlink()
                raise RuntimeError(
                    f"Couldn't download the YuNet face model ({e}). "
                    "Check your network, or use --face-backend mediapipe "
                    "(pip install mediapipe)."
                ) from e
        self.det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        self.max_side = max_side

    def has_face(self, path: Path) -> bool:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return False
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return False
            h, w = img.shape[:2]
            scale = self.max_side / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                h, w = img.shape[:2]
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
            return faces is not None and len(faces) > 0
        except Exception as e:
            logger.debug("YuNet face detect failed on %s: %s", path, e)
            return False


def make_face_detector(backend: str):
    """Build a face detector.

    backend: 'mediapipe', 'yunet', 'haar', 'auto', or 'off'.
    'auto' tries MediaPipe -> YuNet -> Haar and uses the first that works.
    """
    if backend == "off":
        return None
    if backend == "mediapipe":
        return MediaPipeFaceDetector()
    if backend == "yunet":
        return YuNetFaceDetector()
    if backend == "haar":
        return HaarFaceDetector()
    if backend != "auto":
        raise ValueError(f"Unknown face backend: {backend}")

    last_err = None
    for name, ctor in (("MediaPipe", MediaPipeFaceDetector),
                       ("YuNet", YuNetFaceDetector),
                       ("Haar", HaarFaceDetector)):
        try:
            det = ctor()
            logger.info("Face backend: %s", name)
            return det
        except Exception as e:
            last_err = e
            logger.info("%s face detector unavailable (%s); trying next", name, e)
    raise RuntimeError(
        f"No face detector available (last error: {last_err}). "
        "Install mediapipe, or upgrade OpenCV to >= 4.5.4."
    )


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

    Results are persisted incrementally (per YOLO batch / face chunk), so an
    interrupted run resumes instead of recomputing everything.

    Returns: {Path: {"has_person": bool, "has_face": bool}}
    """
    # Build the face detector FIRST: a broken/missing backend should fail in
    # seconds, not after a long YOLO pass.
    face_detector = make_face_detector(face_backend)

    results: dict[Path, dict] = {}

    # 1) Partition: fully cached / person-known (resume) / needs YOLO
    need_yolo: list[Path] = []
    need_face: list[Path] = []
    person_resumed: list[Path] = []
    for p in files:
        cached = cache.get_detection(p)
        if cached is not None:
            results[p] = {"has_person": cached[0], "has_face": cached[1]}
            continue
        person = cache.get_person(p)
        if person is None:
            need_yolo.append(p)
        elif person:
            # Person present — face check is skipped for these anyway.
            results[p] = {"has_person": True, "has_face": False}
            person_resumed.append(p)
        else:
            need_face.append(p)

    # Complete rows for resumed person-hits so future runs take the fast path.
    if person_resumed:
        with cache.transaction():
            for p in person_resumed:
                cache.set_detection(p, True, False)

    if not need_yolo and not need_face:
        logger.info("Detection: all %d files served from cache.", len(files))
        return results

    logger.info(
        "Detection: %d fully cached, %d resumed mid-run, %d to face-check, "
        "%d need full processing",
        len(results) - len(person_resumed), len(person_resumed),
        len(need_face), len(need_yolo),
    )

    # 2) YOLO person detection — person flags are cached per batch.
    if need_yolo and not skip_yolo:
        yolo = load_yolo(device)
        ydev = yolo_device_arg(device)
        try:
            pbar = tqdm(total=len(need_yolo), desc="YOLO person", unit="img")
            for i in range(0, len(need_yolo), yolo_batch):
                chunk = need_yolo[i:i + yolo_batch]
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

                with cache.transaction():
                    for p, res in zip(chunk, res_list):
                        has_person = False
                        try:
                            if res is not None and getattr(res, "boxes", None) is not None:
                                has_person = len(res.boxes) > 0
                        except Exception:
                            has_person = False
                        if has_person:
                            results[p] = {"has_person": True, "has_face": False}
                            cache.set_detection(p, True, False)
                        else:
                            need_face.append(p)
                            cache.set_person(p, False)
                pbar.update(len(chunk))
            pbar.close()
        finally:
            # Free model memory between stages
            del yolo
    elif need_yolo:
        # YOLO explicitly skipped: treat as no person, still face-check.
        with cache.transaction():
            for p in need_yolo:
                cache.set_person(p, False)
        need_face.extend(need_yolo)

    # 3) Face detection on files without a person — cached in chunks.
    if face_detector is None:
        with cache.transaction():
            for p in need_face:
                results[p] = {"has_person": False, "has_face": False}
                cache.set_detection(p, False, False)
    elif need_face:
        backend_name = type(face_detector).__name__.replace("FaceDetector", "")
        pbar = tqdm(total=len(need_face), desc=f"Face detect ({backend_name})",
                    unit="img")
        chunk_size = 128
        for i in range(0, len(need_face), chunk_size):
            chunk = need_face[i:i + chunk_size]
            flags = [(p, face_detector.has_face(p)) for p in chunk]
            with cache.transaction():
                for p, has_face in flags:
                    results[p] = {"has_person": False, "has_face": has_face}
                    cache.set_detection(p, False, has_face)
            pbar.update(len(chunk))
        pbar.close()

    n_person = sum(1 for v in results.values() if v["has_person"])
    n_face = sum(1 for v in results.values()
                 if not v["has_person"] and v["has_face"])
    logger.info(
        "Detection complete: %d people, %d face-only, %d clean (of %d total)",
        n_person, n_face, len(files) - n_person - n_face, len(files),
    )
    return results

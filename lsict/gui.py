"""LSICT local web GUI (gradio). Launch with `lsict gui`.

Serves on 127.0.0.1 only by default — images never leave the machine.
Jobs run in a worker thread; log lines from the pipeline's loggers stream
into the page while the job runs.
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
from pathlib import Path

from lsict import __version__
from lsict.core import is_image_file, normalize_path
from lsict.pipeline import OUTCOMES_CSV, run_full_pipeline, run_nsfw_screen

logger = logging.getLogger(__name__)

MAX_GALLERY_ITEMS = 400
MAX_LOG_LINES = 500

DISCLAIMER = (
    "⚠️ **No guarantees** — filtering is model-based and imperfect. "
    "Manually review the final set (Review tab) before using it anywhere sensitive."
)


# ---------- job runner with live log streaming ----------

class _QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


_job_lock = threading.Lock()


def _stream_job(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread; yield (log_text, summary) tuples.

    summary is None until the job finishes. Only one job runs at a time.
    """
    if not _job_lock.acquire(blocking=False):
        yield "Another job is already running — wait for it to finish.", None
        return

    q: queue.Queue = queue.Queue()
    handler = _QueueLogHandler(q)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    result: dict = {}
    error: list[BaseException] = []

    def target():
        try:
            result["summary"] = fn(*args, **kwargs)
        except BaseException as e:  # surface everything in the UI
            error.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()

    lines: list[str] = []
    try:
        while True:
            drained = False
            try:
                while True:
                    lines.append(q.get_nowait())
                    drained = True
            except queue.Empty:
                pass
            if len(lines) > MAX_LOG_LINES:
                lines = lines[-MAX_LOG_LINES:]
            if not t.is_alive() and not drained:
                break
            yield "\n".join(lines), None
            t.join(timeout=0.5)
    finally:
        root.removeHandler(handler)
        _job_lock.release()

    if error:
        lines.append(f"ERROR: {error[0]}")
        yield "\n".join(lines), {"error": str(error[0])}
    else:
        lines.append("Done.")
        yield "\n".join(lines), result.get("summary")


# ---------- tab callbacks ----------

def _parse_dirs(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _curate_job(inputs_text, kept, unkept, copy_instead,
                device, face_backend, use_faiss, skip_detect,
                similarity, phash_hamming, yolo_conf, rep_policy,
                kept_side, jpeg_quality, clean_kept, no_cache):
    input_dirs = _parse_dirs(inputs_text)
    if not input_dirs:
        yield "Add at least one input folder (one per line).", None
        return
    if not (kept or "").strip() or not (unkept or "").strip():
        yield "Both a Kept folder and an Unkept folder are required.", None
        return
    yield from _stream_job(
        run_full_pipeline,
        input_dirs,
        kept.strip(),
        unkept.strip(),
        copy_instead=bool(copy_instead),
        device=device,
        face_backend=face_backend,
        use_faiss=bool(use_faiss),
        skip_detect=bool(skip_detect),
        similarity=float(similarity),
        phash_hamming=int(phash_hamming),
        yolo_conf=float(yolo_conf),
        rep_policy=rep_policy,
        kept_side=int(kept_side),
        jpeg_quality=int(jpeg_quality),
        clean_kept=bool(clean_kept),
        no_cache=bool(no_cache),
    )


def _nsfw_job(src, dst, backend, threshold, copy, device, batch):
    if not (src or "").strip() or not (dst or "").strip():
        yield "Both a source folder and a destination folder are required.", None
        return
    yield from _stream_job(
        run_nsfw_screen,
        src.strip(),
        dst.strip(),
        backend=backend,
        threshold=float(threshold),
        copy=bool(copy),
        device=device,
        batch=int(batch),
    )


def _numeric_stem_key(p: Path):
    try:
        return (0, int(p.stem))
    except ValueError:
        return (1, p.stem)


def _load_review(kept, unkept):
    """Build the review galleries: kept exports and rejected files with reasons."""
    kept_items: list[tuple[str, str]] = []
    unkept_items: list[tuple[str, str]] = []
    notes: list[str] = []

    kept = (kept or "").strip()
    unkept = (unkept or "").strip()

    n_kept = 0
    if kept:
        kd = normalize_path(kept)
        if kd.exists():
            files = sorted(
                (p for p in kd.iterdir() if p.is_file() and is_image_file(p)),
                key=_numeric_stem_key,
            )
            n_kept = len(files)
            kept_items = [(str(p), p.name) for p in files[:MAX_GALLERY_ITEMS]]
        else:
            notes.append(f"Kept folder not found: {kd}")

    n_unkept = 0
    if unkept:
        ud = normalize_path(unkept)
        if ud.exists():
            reasons: dict[str, str] = {}
            oc = ud / OUTCOMES_CSV
            if oc.exists():
                try:
                    with open(oc, newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            reasons[row["rel_path"]] = row["reason"]
                except Exception as e:
                    notes.append(f"Couldn't read {OUTCOMES_CSV}: {e}")
            else:
                notes.append(
                    f"No {OUTCOMES_CSV} in the Unkept folder — rejection reasons "
                    "appear after a Curate run."
                )
            files = sorted(p for p in ud.rglob("*")
                           if p.is_file() and is_image_file(p))
            n_unkept = len(files)
            for p in files[:MAX_GALLERY_ITEMS]:
                rel = p.relative_to(ud).as_posix()
                unkept_items.append((str(p), reasons.get(rel, rel)))
        else:
            notes.append(f"Unkept folder not found: {ud}")

    status = f"**Kept: {n_kept}** &nbsp;&nbsp;|&nbsp;&nbsp; **Rejected: {n_unkept}**"
    if n_kept > MAX_GALLERY_ITEMS or n_unkept > MAX_GALLERY_ITEMS:
        status += f" — showing the first {MAX_GALLERY_ITEMS} of each"
    if notes:
        status += "<br>" + "<br>".join(notes)
    return status, kept_items, unkept_items


# ---------- UI ----------

def build_ui():
    import gradio as gr

    with gr.Blocks(title="LSICT") as demo:
        gr.Markdown(f"# LSICT\nLarge-scale image curation — v{__version__}\n\n{DISCLAIMER}")

        # ----- Curate -----
        with gr.Tab("Curate"):
            with gr.Row():
                with gr.Column(scale=1):
                    inputs_text = gr.Textbox(
                        label="Input folders (one per line)", lines=3,
                        placeholder="/path/to/photos",
                    )
                    kept = gr.Textbox(label="Kept folder (exports go here)")
                    unkept = gr.Textbox(label="Unkept folder (rejects mirrored here)")
                    copy_instead = gr.Checkbox(
                        value=True,
                        label="Copy rejects instead of moving them (recommended)",
                    )
                    use_faiss = gr.Checkbox(
                        value=False, label="Use FAISS (faster on >50k images)")
                    with gr.Accordion("Advanced", open=False):
                        device = gr.Dropdown(
                            ["auto", "cuda", "mps", "cpu"], value="auto",
                            label="Device")
                        face_backend = gr.Dropdown(
                            ["auto", "mediapipe", "haar", "off"], value="auto",
                            label="Face detector")
                        similarity = gr.Slider(
                            0.70, 1.00, value=0.90, step=0.01,
                            label="Near-duplicate similarity (higher = stricter)")
                        phash_hamming = gr.Slider(
                            0, 100, value=30, step=1,
                            label="pHash Hamming threshold (ignored with FAISS)")
                        yolo_conf = gr.Slider(
                            0.05, 0.90, value=0.25, step=0.05,
                            label="Person-detection confidence")
                        rep_policy = gr.Dropdown(
                            ["sharpest", "largest", "newest", "oldest", "first"],
                            value="sharpest", label="Keep which duplicate")
                        kept_side = gr.Number(value=300, precision=0,
                                              label="Export size (px)")
                        jpeg_quality = gr.Slider(50, 100, value=92, step=1,
                                                 label="JPEG quality")
                        skip_detect = gr.Checkbox(
                            value=False, label="Skip people/face detection")
                        clean_kept = gr.Checkbox(
                            value=True,
                            label="Clear existing exports in Kept first")
                        no_cache = gr.Checkbox(
                            value=False, label="Disable cache (recompute everything)")
                    run_btn = gr.Button("Run pipeline", variant="primary")
                with gr.Column(scale=1):
                    run_log = gr.Textbox(label="Log", lines=22, interactive=False)
                    run_summary = gr.JSON(label="Summary")
            run_btn.click(
                _curate_job,
                inputs=[inputs_text, kept, unkept, copy_instead,
                        device, face_backend, use_faiss, skip_detect,
                        similarity, phash_hamming, yolo_conf, rep_policy,
                        kept_side, jpeg_quality, clean_kept, no_cache],
                outputs=[run_log, run_summary],
            )

        # ----- NSFW screen -----
        with gr.Tab("NSFW screen"):
            with gr.Row():
                with gr.Column(scale=1):
                    nsfw_src = gr.Textbox(label="Source folder (e.g. your Kept folder)")
                    nsfw_dst = gr.Textbox(label="Destination for SAFE images")
                    nsfw_copy = gr.Checkbox(value=True, label="Copy instead of move")
                    nsfw_backend = gr.Dropdown(
                        ["auto", "classifier", "clip"], value="auto", label="Backend")
                    nsfw_threshold = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05,
                        label="NSFW threshold (lower = stricter)")
                    with gr.Accordion("Advanced", open=False):
                        nsfw_device = gr.Dropdown(
                            ["auto", "cuda", "mps", "cpu"], value="auto",
                            label="Device")
                        nsfw_batch = gr.Number(value=32, precision=0, label="Batch size")
                    nsfw_btn = gr.Button("Screen", variant="primary")
                with gr.Column(scale=1):
                    nsfw_log = gr.Textbox(label="Log", lines=16, interactive=False)
                    nsfw_summary = gr.JSON(label="Summary")
            nsfw_btn.click(
                _nsfw_job,
                inputs=[nsfw_src, nsfw_dst, nsfw_backend, nsfw_threshold,
                        nsfw_copy, nsfw_device, nsfw_batch],
                outputs=[nsfw_log, nsfw_summary],
            )

        # ----- Review -----
        with gr.Tab("Review"):
            gr.Markdown(
                "Check the model's work: kept exports on the left, rejected "
                "files (with the reason) on the right."
            )
            with gr.Row():
                review_kept = gr.Textbox(label="Kept folder")
                review_unkept = gr.Textbox(label="Unkept folder")
                review_btn = gr.Button("Load", variant="primary")
            review_status = gr.Markdown()
            with gr.Row():
                kept_gallery = gr.Gallery(
                    label="Kept", columns=6, height=560, object_fit="contain")
                unkept_gallery = gr.Gallery(
                    label="Rejected (caption = reason)", columns=6, height=560,
                    object_fit="contain")
            review_btn.click(
                _load_review,
                inputs=[review_kept, review_unkept],
                outputs=[review_status, kept_gallery, unkept_gallery],
            )

    return demo


def launch_gui(host: str = "127.0.0.1", port: int = 7860,
               open_browser: bool = True) -> None:
    demo = build_ui()
    logger.info("LSICT GUI on http://%s:%d", host, port)
    demo.queue().launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
    )

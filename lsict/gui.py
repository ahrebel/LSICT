"""LSICT local web GUI (gradio). Launch with `lsict gui`.

Serves on 127.0.0.1 only by default — images never leave the machine.
Jobs run in a worker thread; log lines from the pipeline's loggers stream
into the page while the job runs, and the pipeline's tqdm progress bars
are mirrored into an on-page progress bar.
"""
from __future__ import annotations

import csv
import html as html_mod
import logging
import queue
import threading
from pathlib import Path

import tqdm as _tqdm_module

from lsict import __version__
from lsict.core import is_image_file, normalize_path
from lsict.pipeline import OUTCOMES_CSV, run_full_pipeline, run_nsfw_screen

logger = logging.getLogger(__name__)

MAX_GALLERY_ITEMS = 400
MAX_LOG_LINES = 500

DISCLAIMER_HTML = (
    '<div id="disclaimer"><strong>No guarantees.</strong> Filtering is '
    "model-based and imperfect — it will occasionally miss people, faces, "
    "duplicates, or unsafe content. Manually review the final set "
    "(<em>Review</em> tab) before using it anywhere sensitive.</div>"
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }

/* ---- header ---- */
#app-header { padding: 6px 2px 0; }
#app-header .brand-row { display: flex; align-items: baseline; gap: 10px; }
#app-header .wordmark {
    font-size: 1.75rem; font-weight: 800; letter-spacing: 0.04em;
    color: var(--body-text-color);
}
#app-header .version-badge {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
    color: var(--primary-600); background: var(--primary-50);
    border: 1px solid var(--primary-200);
    border-radius: 999px; padding: 2px 10px; transform: translateY(-2px);
}
.dark #app-header .version-badge {
    color: var(--primary-300); background: rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.35);
}
#app-header .tagline {
    margin: 2px 0 0; font-size: 0.92rem;
    color: var(--body-text-color-subdued);
}

/* ---- disclaimer banner ---- */
#disclaimer {
    margin: 12px 0 4px; padding: 10px 14px; font-size: 0.88rem;
    line-height: 1.45; border-radius: 8px;
    border: 1px solid rgba(217, 119, 6, 0.35);
    border-left: 4px solid #d97706;
    background: rgba(245, 158, 11, 0.08);
    color: var(--body-text-color);
}

/* ---- section headings inside tabs ---- */
.section-title h4 {
    margin: 2px 0 0; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--body-text-color-subdued);
}

/* ---- job progress bar ---- */
.prog { margin: 0 0 8px; }
.prog-label {
    font-size: 0.82rem; font-weight: 600; margin-bottom: 5px;
    color: var(--body-text-color); font-variant-numeric: tabular-nums;
}
.prog-track {
    height: 8px; border-radius: 999px; overflow: hidden;
    background: rgba(100, 116, 139, 0.22);
}
.prog-fill {
    height: 100%; border-radius: 999px; background: #6366f1;
    transition: width 0.4s ease;
}
.prog-indeterminate {
    width: 30% !important;
    animation: prog-slide 1.2s ease-in-out infinite alternate;
}
@keyframes prog-slide {
    from { margin-left: 0; }
    to   { margin-left: 70%; }
}

/* ---- console-style log ---- */
.log-box textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
    font-size: 12px !important; line-height: 1.55 !important;
    background: #0b1220 !important; color: #d5e0f0 !important;
    border-radius: 8px !important;
}

/* ---- full-width action buttons ---- */
.action-btn { width: 100%; }

/* ---- footer ---- */
#app-footer {
    text-align: center; padding: 20px 0 8px; font-size: 0.78rem;
    color: var(--body-text-color-subdued);
}
"""


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


class _TqdmHook:
    """Mirror the most recently updated tqdm bar's state while active.

    The pipeline modules use tqdm for every stage; patching update/close on
    the tqdm class lets the GUI show the same progress without touching them.
    """

    def __init__(self):
        self.desc = ""
        self.n = 0
        self.total: int | None = None
        self._orig_update = None
        self._orig_close = None

    def _capture(self, bar) -> None:
        self.desc = bar.desc or ""
        self.n = int(bar.n)
        self.total = int(bar.total) if bar.total else None

    def __enter__(self) -> "_TqdmHook":
        hook = self
        cls = _tqdm_module.tqdm
        self._orig_update = cls.update
        self._orig_close = cls.close

        def update(bar, n=1):
            res = hook._orig_update(bar, n)
            hook._capture(bar)
            return res

        def close(bar):
            hook._capture(bar)
            return hook._orig_close(bar)

        cls.update = update
        cls.close = close
        return self

    def __exit__(self, *exc) -> None:
        cls = _tqdm_module.tqdm
        cls.update = self._orig_update
        cls.close = self._orig_close

    def as_html(self) -> str:
        if not self.desc and not self.n:
            return ""
        desc = html_mod.escape(self.desc) or "Working"
        if self.total:
            pct = min(100.0, 100.0 * self.n / self.total)
            label = f"{desc} — {self.n:,} / {self.total:,} ({pct:.0f}%)"
            fill = f'<div class="prog-fill" style="width:{pct:.1f}%"></div>'
        else:
            label = f"{desc} — {self.n:,}"
            fill = '<div class="prog-fill prog-indeterminate"></div>'
        return (
            f'<div class="prog"><div class="prog-label">{label}</div>'
            f'<div class="prog-track">{fill}</div></div>'
        )


def _stream_job(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread.

    Yields (log_text, progress_html, summary) tuples; summary is None until
    the job finishes. Only one job runs at a time.
    """
    if not _job_lock.acquire(blocking=False):
        yield "Another job is already running — wait for it to finish.", "", None
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

    lines: list[str] = []
    try:
        with _TqdmHook() as hook:
            t = threading.Thread(target=target, daemon=True)
            t.start()
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
                yield "\n".join(lines), hook.as_html(), None
                t.join(timeout=0.5)
    finally:
        root.removeHandler(handler)
        _job_lock.release()

    if error:
        lines.append(f"ERROR: {error[0]}")
        yield "\n".join(lines), "", {"error": str(error[0])}
    else:
        lines.append("Done.")
        yield "\n".join(lines), "", result.get("summary")


# ---------- tab callbacks ----------

def _parse_dirs(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _curate_job(inputs_text, kept, unkept, copy_instead,
                device, face_backend, use_faiss, skip_detect,
                similarity, phash_hamming, yolo_conf, rep_policy,
                kept_side, jpeg_quality, clean_kept, no_cache):
    input_dirs = _parse_dirs(inputs_text)
    if not input_dirs:
        yield "Add at least one input folder (one per line).", "", None
        return
    if not (kept or "").strip() or not (unkept or "").strip():
        yield "Both a Kept folder and an Unkept folder are required.", "", None
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
        yield "Both a source folder and a destination folder are required.", "", None
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

def _make_theme(gr):
    return gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_md,
        font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI",
              "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas",
                   "monospace"],
    )


def build_ui():
    import gradio as gr

    with gr.Blocks(title="LSICT") as demo:
        gr.HTML(
            '<div id="app-header">'
            '  <div class="brand-row">'
            '    <span class="wordmark">LSICT</span>'
            f'    <span class="version-badge">v{__version__}</span>'
            "  </div>"
            '  <p class="tagline">Large-scale image curation — people/face '
            "filtering, deduplication, square-crop export, NSFW screening. "
            "Runs entirely on this machine.</p>"
            "</div>"
            f"{DISCLAIMER_HTML}"
        )

        # ----- Curate -----
        with gr.Tab("Curate"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("#### Folders", elem_classes=["section-title"])
                    with gr.Group():
                        inputs_text = gr.Textbox(
                            label="Input folders (one per line)", lines=3,
                            placeholder="/path/to/photos",
                        )
                        kept = gr.Textbox(
                            label="Kept folder",
                            info="Numbered 300×300 exports go here",
                            placeholder="/path/to/Kept",
                        )
                        unkept = gr.Textbox(
                            label="Unkept folder",
                            info="Rejected files are mirrored here",
                            placeholder="/path/to/Unkept",
                        )
                    gr.Markdown("#### Options", elem_classes=["section-title"])
                    with gr.Group():
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
                    run_btn = gr.Button("Run pipeline", variant="primary",
                                        size="lg", elem_classes=["action-btn"])
                with gr.Column(scale=1):
                    run_progress = gr.HTML()
                    run_log = gr.Textbox(
                        label="Log", lines=22, interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Pipeline output appears here…")
                    run_summary = gr.JSON(label="Summary")
            run_btn.click(
                _curate_job,
                inputs=[inputs_text, kept, unkept, copy_instead,
                        device, face_backend, use_faiss, skip_detect,
                        similarity, phash_hamming, yolo_conf, rep_policy,
                        kept_side, jpeg_quality, clean_kept, no_cache],
                outputs=[run_log, run_progress, run_summary],
            )

        # ----- NSFW screen -----
        with gr.Tab("NSFW screen"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("#### Folders", elem_classes=["section-title"])
                    with gr.Group():
                        nsfw_src = gr.Textbox(
                            label="Source folder",
                            info="Usually your Kept folder",
                            placeholder="/path/to/Kept")
                        nsfw_dst = gr.Textbox(
                            label="Destination for SAFE images",
                            placeholder="/path/to/FinalImageSet")
                    gr.Markdown("#### Options", elem_classes=["section-title"])
                    with gr.Group():
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
                    nsfw_btn = gr.Button("Screen", variant="primary",
                                         size="lg", elem_classes=["action-btn"])
                with gr.Column(scale=1):
                    nsfw_progress = gr.HTML()
                    nsfw_log = gr.Textbox(
                        label="Log", lines=16, interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Screening output appears here…")
                    nsfw_summary = gr.JSON(label="Summary")
            nsfw_btn.click(
                _nsfw_job,
                inputs=[nsfw_src, nsfw_dst, nsfw_backend, nsfw_threshold,
                        nsfw_copy, nsfw_device, nsfw_batch],
                outputs=[nsfw_log, nsfw_progress, nsfw_summary],
            )

        # ----- Review -----
        with gr.Tab("Review"):
            gr.Markdown(
                "Check the model's work: kept exports on the left, rejected "
                "files (with the reason) on the right."
            )
            with gr.Row():
                review_kept = gr.Textbox(label="Kept folder", scale=2,
                                         placeholder="/path/to/Kept")
                review_unkept = gr.Textbox(label="Unkept folder", scale=2,
                                           placeholder="/path/to/Unkept")
                review_btn = gr.Button("Load", variant="primary", scale=1)
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

        gr.HTML(
            f'<div id="app-footer">LSICT v{__version__} · MIT license · '
            "local only — nothing leaves this machine</div>"
        )

    return demo


def launch_gui(host: str = "127.0.0.1", port: int = 7860,
               open_browser: bool = True, **launch_kwargs) -> None:
    import gradio as gr

    demo = build_ui()
    logger.info("LSICT GUI on http://%s:%d", host, port)
    demo.queue().launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
        theme=_make_theme(gr),
        css=CSS,
        **launch_kwargs,
    )

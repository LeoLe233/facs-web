from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from facs_anime_analysis import (
    IMAGE_EXTENSIONS,
    configure_windows_ffmpeg_dlls,
    describe_au,
    emotion_columns,
    normalize_au_code,
)


BASE_DIR = Path(__file__).resolve().parent
WEB_RUNS_DIR = BASE_DIR / "web_runs"
UPLOAD_DIR = WEB_RUNS_DIR / "uploads"
RESULTS_DIR = WEB_RUNS_DIR / "results"
PYFEAT_RESULTS_DIR = BASE_DIR / "py-feat_results"
ALLOWED_EXTENSIONS = {extension.lstrip(".") for extension in IMAGE_EXTENSIONS}
WEB_BACKENDS = {"pyfeat": "DetectorV1", "pyfeat-v2": "DetectorV2"}
WEB_PYFEAT_BATCH_SIZE = max(1, int(os.environ.get("FACS_WEB_PYFEAT_BATCH_SIZE", "4")))
WEB_PYFEAT_OUTPUT_SIZE = max(0, int(os.environ.get("FACS_WEB_PYFEAT_OUTPUT_SIZE", "640")))
MAX_ANALYSIS_WORKERS = max(1, int(os.environ.get("FACS_WEB_MAX_WORKERS", "2")))
ProgressCallback = Callable[[int, int], None]


def default_analyzer_python() -> Path:
    executable = "python.exe" if sys.platform == "win32" else "python"
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    return BASE_DIR / ".venv311" / scripts_dir / executable


ANALYZER_PYTHON = Path(os.environ.get("FACS_ANALYZER_PYTHON", default_analyzer_python()))


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-facs-demo")
analysis_executor = ThreadPoolExecutor(max_workers=MAX_ANALYSIS_WORKERS)
analysis_jobs: dict[str, dict[str, object]] = {}
analysis_jobs_lock = Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def static_result_url(run_id: str, relative_path: str) -> str:
    return f"/results/{run_id}/{relative_path}"


def static_upload_url(run_id: str, filename: str) -> str:
    return f"/uploads/{run_id}/{filename}"


def row_records(df: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    if df.empty:
        return []
    clean = df.head(limit).copy()
    clean = clean.astype(object)
    clean[pd.isna(clean)] = None
    records = clean.to_dict(orient="records")
    return [{str(key): value for key, value in record.items()} for record in records]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_style_comparison_data() -> dict[str, object]:
    emotion_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    preferred_order = ["original", "disney", "ghibli"]
    style_dirs = {
        path.name.lower(): path
        for path in PYFEAT_RESULTS_DIR.iterdir()
        if path.is_dir()
    } if PYFEAT_RESULTS_DIR.exists() else {}
    ordered_names = [name for name in preferred_order if name in style_dirs]
    ordered_names.extend(sorted(name for name in style_dirs if name not in preferred_order))

    styles = []
    au_descriptions: dict[str, str] = {}
    for style_name in ordered_names:
        style_dir = style_dirs[style_name]
        emotion_summary = read_csv_if_exists(style_dir / "emotion_summary.csv")
        emotion_predictions = read_csv_if_exists(style_dir / "emotion_predictions.csv")
        au_summary = read_csv_if_exists(style_dir / "au_summary.csv")
        detection_summary = read_csv_if_exists(style_dir / "detection_summary.csv")
        au_reference = read_csv_if_exists(style_dir / "au_reference.csv")

        if not au_reference.empty and {"au", "description"}.issubset(au_reference.columns):
            au_descriptions.update(
                {
                    str(row["au"]): str(row["description"])
                    for _, row in au_reference.iterrows()
                }
            )

        emotion_means = {
            emotion: float(emotion_summary.iloc[0].get(emotion, 0.0)) if not emotion_summary.empty else 0.0
            for emotion in emotion_names
        }
        prediction_counts = (
            emotion_predictions["top_emotion"].astype(str).str.lower().value_counts()
            if "top_emotion" in emotion_predictions.columns
            else pd.Series(dtype=int)
        )
        prediction_total = int(prediction_counts.sum())
        top_emotions = {
            emotion: {
                "count": int(prediction_counts.get(emotion, 0)),
                "rate": float(prediction_counts.get(emotion, 0) / prediction_total) if prediction_total else 0.0,
            }
            for emotion in emotion_names
        }
        au_values = {
            str(column): float(au_summary.iloc[0][column])
            for column in au_summary.columns
            if str(column).startswith("AU") and pd.notna(au_summary.iloc[0][column])
        } if not au_summary.empty else {}
        detected = int(detection_summary.iloc[0].get("detected", prediction_total)) if not detection_summary.empty else prediction_total
        images = int(detection_summary.iloc[0].get("images", prediction_total)) if not detection_summary.empty else prediction_total

        styles.append(
            {
                "id": style_name,
                "label": style_name.title(),
                "images": images,
                "detected": detected,
                "detection_rate": float(detected / images) if images else 0.0,
                "emotion_means": emotion_means,
                "top_emotions": top_emotions,
                "au_values": au_values,
            }
        )

    return {
        "styles": styles,
        "emotions": emotion_names,
        "au_descriptions": au_descriptions,
    }


def run_batch_analysis(
    run_id: str,
    backend: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, float]:
    run_input_dir = UPLOAD_DIR / run_id
    run_output_dir = RESULTS_DIR / run_id
    run_input_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if not ANALYZER_PYTHON.exists():
        raise RuntimeError(
            f"Analyzer Python not found at {ANALYZER_PYTHON}. "
            "Run the app with .venv311, or set FACS_ANALYZER_PYTHON to the Python that has Py-Feat installed."
        )

    command = [
        str(ANALYZER_PYTHON),
        str(BASE_DIR / "facs_anime_analysis.py"),
        "--backend",
        backend,
        "--input-dir",
        str(run_input_dir),
        "--output-dir",
        str(run_output_dir),
        "--group-by",
        "style",
        "--pyfeat-batch-size",
        str(WEB_PYFEAT_BATCH_SIZE),
    ]
    if WEB_PYFEAT_OUTPUT_SIZE:
        command.extend(["--pyfeat-output-size", str(WEB_PYFEAT_OUTPUT_SIZE)])

    analyzer_env = os.environ.copy()
    ffmpeg_dll_dirs = configure_windows_ffmpeg_dlls()
    if ffmpeg_dll_dirs and not analyzer_env.get("FACS_FFMPEG_DLL_DIR"):
        analyzer_env["FACS_FFMPEG_DLL_DIR"] = os.pathsep.join(str(path) for path in ffmpeg_dll_dirs)

    process_started_at = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=analyzer_env,
    )
    output_tail: deque[str] = deque(maxlen=80)
    if process.stdout is not None:
        for line in process.stdout:
            output_tail.append(line.rstrip())
            if line.startswith("FACS_PROGRESS "):
                parts = line.split()
                if len(parts) == 3 and progress_callback is not None:
                    try:
                        progress_callback(int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
    return_code = process.wait()
    process_runtime_seconds = time.perf_counter() - process_started_at
    if return_code != 0 and not (run_output_dir / "au_results.csv").exists():
        details = "\n".join(output_tail).strip() or "unknown error"
        raise RuntimeError(details[-1200:])

    return run_output_dir, process_runtime_seconds


def result_rows_for_image(results: pd.DataFrame, image_path: Path) -> pd.DataFrame:
    if results.empty:
        return results
    if "filename" in results.columns:
        return results[results["filename"].astype(str) == image_path.name].copy()
    if "image_path" in results.columns:
        return results[results["image_path"].astype(str).map(lambda value: Path(value).name) == image_path.name].copy()
    return pd.DataFrame()


def is_success_value(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def first_non_empty(df: pd.DataFrame, column: str, default: object = None) -> object:
    if column not in df.columns:
        return default
    values = df[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def build_upload_result(
    image_path: Path,
    run_id: str,
    backend: str,
    run_output_dir: Path,
    all_results: pd.DataFrame,
    au_reference: pd.DataFrame,
    process_runtime_seconds: float,
) -> dict[str, object]:
    results = result_rows_for_image(all_results, image_path)
    detection_summary = read_csv_if_exists(run_output_dir / "detection_summary.csv")
    au_summary = read_csv_if_exists(run_output_dir / "au_summary.csv")
    emotion_predictions = read_csv_if_exists(run_output_dir / "emotion_predictions.csv")
    emotion_summary = read_csv_if_exists(run_output_dir / "emotion_summary.csv")

    emotion_cols = emotion_columns(results)
    detections = results.copy()
    detections["face_index"] = detections.groupby("image_path").cumcount() if "image_path" in detections else 0
    detections = detections.astype(object)
    detections[pd.isna(detections)] = None

    face_cards = []
    for _, row in detections.iterrows():
        emotions = []
        if emotion_cols:
            ranked = row[emotion_cols].dropna().sort_values(ascending=False)
            emotions = [
                {"name": emotion, "score": float(score)}
                for emotion, score in ranked.head(7).items()
            ]

        au_cols = [
            col
            for col in results.columns
            if col.startswith("AU") and pd.api.types.is_numeric_dtype(results[col])
        ]
        aus = []
        for col in au_cols:
            value = row[col]
            if value is not None:
                code = normalize_au_code(col)
                aus.append(
                    {
                        "name": code,
                        "column": col,
                        "description": describe_au(code),
                        "score": float(value),
                    }
                )
        aus = sorted(aus, key=lambda item: item["score"], reverse=True)[:10]

        face_cards.append(
            {
                "face_index": int(row["face_index"]),
                "success": is_success_value(row.get("success")),
                "emotions": emotions,
                "aus": aus,
            }
        )

    overlay_paths = sorted((run_output_dir / "au_overlays").glob(f"{image_path.stem}_face*_au_overlay.png"))
    plot_paths = {
        path.stem: static_result_url(run_id, f"plots/{path.name}")
        for path in sorted((run_output_dir / "plots").glob("*.png"))
    }

    return {
        "run_id": run_id,
        "filename": image_path.name,
        "backend": WEB_BACKENDS.get(backend, backend),
        "accelerator": first_non_empty(results, "pyfeat_device", "unknown"),
        "process_seconds": round(process_runtime_seconds, 2),
        "pyfeat_seconds": round(float(first_non_empty(results, "pyfeat_runtime_seconds", 0.0)), 2),
        "batch_size": WEB_PYFEAT_BATCH_SIZE,
        "original_url": static_upload_url(run_id, image_path.name),
        "overlay_urls": [static_result_url(run_id, f"au_overlays/{path.name}") for path in overlay_paths],
        "plot_urls": plot_paths,
        "detection_summary": row_records(detection_summary),
        "au_reference": row_records(au_reference, limit=32),
        "emotion_predictions": row_records(emotion_predictions),
        "emotion_summary": row_records(emotion_summary),
        "au_summary": row_records(au_summary),
        "face_cards": face_cards,
        "downloads": {
            "AU results": static_result_url(run_id, "au_results.csv"),
            "AU reference": static_result_url(run_id, "au_reference.csv") if (run_output_dir / "au_reference.csv").exists() else None,
            "Detection summary": static_result_url(run_id, "detection_summary.csv"),
            "AU summary": static_result_url(run_id, "au_summary.csv") if (run_output_dir / "au_summary.csv").exists() else None,
            "Emotion predictions": static_result_url(run_id, "emotion_predictions.csv") if (run_output_dir / "emotion_predictions.csv").exists() else None,
            "Emotion summary": static_result_url(run_id, "emotion_summary.csv") if (run_output_dir / "emotion_summary.csv").exists() else None,
        },
    }


def failed_result(image_path: Path, error: str, backend: str = "") -> dict[str, object]:
    return {
        "run_id": "",
        "filename": image_path.name,
        "backend": WEB_BACKENDS.get(backend, backend),
        "original_url": "",
        "overlay_urls": [],
        "plot_urls": {},
        "detection_summary": [],
        "au_reference": [],
        "emotion_predictions": [],
        "emotion_summary": [],
        "au_summary": [],
        "face_cards": [],
        "downloads": {},
        "error": error,
    }


def analyze_upload_batch(
    image_paths: list[Path],
    batch_id: str,
    backend: str,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, object]]:
    try:
        run_output_dir, process_runtime_seconds = run_batch_analysis(batch_id, backend, progress_callback)
        all_results = read_csv_if_exists(run_output_dir / "au_results.csv")
        au_reference = read_csv_if_exists(run_output_dir / "au_reference.csv")
        return [
            build_upload_result(
                image_path,
                batch_id,
                backend,
                run_output_dir,
                all_results,
                au_reference,
                process_runtime_seconds,
            )
            for image_path in image_paths
        ]
    except Exception as exc:
        return [failed_result(image_path, str(exc), backend) for image_path in image_paths]


def save_uploads(uploads: list[FileStorage]) -> tuple[str, list[Path]]:
    batch_id = uuid.uuid4().hex[:12]
    run_input_dir = UPLOAD_DIR / batch_id
    run_input_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    used_stems: set[str] = set()
    for index, uploaded in enumerate(uploads):
        filename = secure_filename(uploaded.filename)
        if not filename:
            filename = f"upload_{index}.png"
        stem = Path(filename).stem or "upload"
        suffix = Path(filename).suffix or ".png"
        if (run_input_dir / filename).exists() or stem.lower() in used_stems:
            filename = f"{stem}_{index:03d}{suffix}"
            stem = Path(filename).stem
        used_stems.add(stem.lower())
        image_path = run_input_dir / filename
        uploaded.save(image_path)
        image_paths.append(image_path)
    return batch_id, image_paths


def set_job_values(job_id: str, **values: object) -> None:
    with analysis_jobs_lock:
        if job_id in analysis_jobs:
            analysis_jobs[job_id].update(values)


def run_analysis_job(job_id: str, image_paths: list[Path], batch_id: str, backend: str) -> None:
    started_at = time.monotonic()
    total = len(image_paths)
    set_job_values(job_id, status="running", message="Loading analysis models…", started_at=started_at)

    def report_progress(completed: int, reported_total: int) -> None:
        completed = min(max(0, completed), total)
        elapsed = max(0.0, time.monotonic() - started_at)
        eta_seconds = (elapsed / completed) * (total - completed) if completed else None
        set_job_values(
            job_id,
            completed=completed,
            total=reported_total or total,
            eta_seconds=round(eta_seconds) if eta_seconds is not None else None,
            message="Analyzing images…" if completed < total else "Preparing results…",
        )

    try:
        results = analyze_upload_batch(image_paths, batch_id, backend, report_progress)
        successful_results = [result for result in results if not result.get("error")]
    except Exception as exc:
        shutil.rmtree(UPLOAD_DIR / batch_id, ignore_errors=True)
        shutil.rmtree(RESULTS_DIR / batch_id, ignore_errors=True)
        set_job_values(job_id, status="failed", message=f"Analysis failed: {exc}", eta_seconds=None)
        return

    if not successful_results:
        error = results[0].get("error") if results else "unknown error"
        shutil.rmtree(UPLOAD_DIR / batch_id, ignore_errors=True)
        shutil.rmtree(RESULTS_DIR / batch_id, ignore_errors=True)
        set_job_values(job_id, status="failed", message=f"Analysis failed: {error}", eta_seconds=None)
        return

    set_job_values(
        job_id,
        status="completed",
        completed=total,
        total=total,
        eta_seconds=0,
        message="Analysis complete.",
        results=results,
    )


def request_uploads() -> list[FileStorage]:
    uploads = [
        uploaded
        for uploaded in request.files.getlist("images")
        if uploaded is not None and uploaded.filename
    ]
    if not uploads:
        single_upload = request.files.get("image")
        if single_upload is not None and single_upload.filename:
            uploads = [single_upload]
    return uploads


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        backend = request.form.get("backend", "pyfeat")
        if backend not in WEB_BACKENDS:
            flash("Choose a supported analysis backend.")
            return redirect(url_for("index"))

        uploads = request_uploads()
        if not uploads:
            flash("Choose at least one image first.")
            return redirect(url_for("index"))

        invalid_names = [uploaded.filename for uploaded in uploads if not allowed_file(uploaded.filename)]
        if invalid_names:
            flash("Upload only PNG, JPG, JPEG, WEBP, BMP, TIF, or TIFF images.")
            return redirect(url_for("index"))

        batch_id, image_paths = save_uploads(uploads)
        run_input_dir = UPLOAD_DIR / batch_id

        results = analyze_upload_batch(image_paths, batch_id, backend)
        successful_results = [result for result in results if not result.get("error")]
        if not successful_results:
            shutil.rmtree(run_input_dir, ignore_errors=True)
            shutil.rmtree(RESULTS_DIR / batch_id, ignore_errors=True)
            flash(f"Analysis failed: {results[0].get('error') if results else 'unknown error'}")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            result=successful_results[0],
            results=results,
            selected_backend=backend,
            backends=WEB_BACKENDS,
        )

    return render_template("index.html", result=None, results=[], selected_backend="pyfeat", backends=WEB_BACKENDS)


@app.post("/analysis/start")
def start_analysis():
    backend = request.form.get("backend", "pyfeat")
    if backend not in WEB_BACKENDS:
        return jsonify(error="Choose a supported analysis backend."), 400

    uploads = request_uploads()
    if not uploads:
        return jsonify(error="Choose at least one image first."), 400

    invalid_names = [uploaded.filename for uploaded in uploads if not allowed_file(uploaded.filename)]
    if invalid_names:
        return jsonify(error="Upload only PNG, JPG, JPEG, WEBP, BMP, TIF, or TIFF images."), 400

    batch_id, image_paths = save_uploads(uploads)
    job_id = uuid.uuid4().hex
    with analysis_jobs_lock:
        analysis_jobs[job_id] = {
            "status": "queued",
            "completed": 0,
            "total": len(image_paths),
            "eta_seconds": None,
            "message": "Waiting to start…",
            "backend": backend,
        }
    analysis_executor.submit(run_analysis_job, job_id, image_paths, batch_id, backend)
    return jsonify(
        job_id=job_id,
        status_url=url_for("analysis_status", job_id=job_id),
        results_url=url_for("analysis_results", job_id=job_id),
    ), 202


@app.get("/analysis/<job_id>/status")
def analysis_status(job_id: str):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            return jsonify(error="Analysis job not found."), 404
        payload = {
            key: job.get(key)
            for key in ("status", "completed", "total", "eta_seconds", "message")
        }
    if payload["status"] == "completed":
        payload["results_url"] = url_for("analysis_results", job_id=job_id)
    return jsonify(payload)


@app.get("/analysis/<job_id>/results")
def analysis_results(job_id: str):
    with analysis_jobs_lock:
        job = analysis_jobs.get(job_id)
        if job is None:
            return redirect(url_for("index"))
        status = job.get("status")
        results = job.get("results")
        backend = str(job.get("backend", "pyfeat"))

    if status != "completed" or not isinstance(results, list):
        return redirect(url_for("index"))
    successful_results = [result for result in results if not result.get("error")]
    return render_template(
        "index.html",
        result=successful_results[0],
        results=results,
        selected_backend=backend,
        backends=WEB_BACKENDS,
    )


@app.get("/style-comparison")
def style_comparison():
    comparison_data = load_style_comparison_data()
    return render_template(
        "comparison.html",
        comparison_data=comparison_data,
        styles=comparison_data["styles"],
    )


@app.route("/uploads/<run_id>/<path:filename>")
def serve_upload_file(run_id: str, filename: str):
    return send_from_directory(UPLOAD_DIR / run_id, filename)


@app.route("/results/<run_id>/<path:filename>")
def serve_result_file(run_id: str, filename: str):
    return send_from_directory(RESULTS_DIR / run_id, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

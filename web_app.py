from __future__ import annotations

import os
import sys
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
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
ALLOWED_EXTENSIONS = {extension.lstrip(".") for extension in IMAGE_EXTENSIONS}
MAX_ANALYSIS_WORKERS = max(1, int(os.environ.get("FACS_WEB_MAX_WORKERS", "2")))


def default_analyzer_python() -> Path:
    executable = "python.exe" if sys.platform == "win32" else "python"
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    return BASE_DIR / ".venv311" / scripts_dir / executable


ANALYZER_PYTHON = Path(os.environ.get("FACS_ANALYZER_PYTHON", default_analyzer_python()))


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-facs-demo")


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


def analyze_upload(image_path: Path, run_id: str) -> dict[str, object]:
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
        "pyfeat",
        "--input-dir",
        str(run_input_dir),
        "--output-dir",
        str(run_output_dir),
        "--group-by",
        "style",
    ]
    analyzer_env = os.environ.copy()
    ffmpeg_dll_dirs = configure_windows_ffmpeg_dlls()
    if ffmpeg_dll_dirs and not analyzer_env.get("FACS_FFMPEG_DLL_DIR"):
        analyzer_env["FACS_FFMPEG_DLL_DIR"] = os.pathsep.join(str(path) for path in ffmpeg_dll_dirs)

    completed = subprocess.run(command, cwd=BASE_DIR, text=True, capture_output=True, env=analyzer_env)
    if completed.returncode != 0 and not (run_output_dir / "au_results.csv").exists():
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(details[-1200:])

    results = read_csv_if_exists(run_output_dir / "au_results.csv")
    detection_summary = read_csv_if_exists(run_output_dir / "detection_summary.csv")
    au_summary = read_csv_if_exists(run_output_dir / "au_summary.csv")
    au_reference = read_csv_if_exists(run_output_dir / "au_reference.csv")
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
                "success": bool(row.get("success")),
                "emotions": emotions,
                "aus": aus,
            }
        )

    overlay_paths = sorted((run_output_dir / "au_overlays").glob("*.png"))
    plot_paths = {
        path.stem: static_result_url(run_id, f"plots/{path.name}")
        for path in sorted((run_output_dir / "plots").glob("*.png"))
    }

    return {
        "run_id": run_id,
        "filename": image_path.name,
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


def failed_result(image_path: Path, error: str) -> dict[str, object]:
    return {
        "run_id": "",
        "filename": image_path.name,
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


def analyze_upload_batch(image_paths: list[Path], batch_id: str) -> list[dict[str, object]]:
    indexed_paths = list(enumerate(image_paths))
    results: list[dict[str, object] | None] = [None] * len(indexed_paths)
    worker_count = min(MAX_ANALYSIS_WORKERS, len(indexed_paths))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(analyze_upload, image_path, f"{batch_id}_{index:03d}"): (index, image_path)
            for index, image_path in indexed_paths
        }
        for future in as_completed(futures):
            index, image_path = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = failed_result(image_path, str(exc))

    return [result for result in results if result is not None]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploads = [
            uploaded
            for uploaded in request.files.getlist("images")
            if uploaded is not None and uploaded.filename
        ]
        if not uploads:
            single_upload = request.files.get("image")
            if single_upload is not None and single_upload.filename:
                uploads = [single_upload]

        if not uploads:
            flash("Choose at least one image first.")
            return redirect(url_for("index"))

        invalid_names = [uploaded.filename for uploaded in uploads if not allowed_file(uploaded.filename)]
        if invalid_names:
            flash("Upload only PNG, JPG, JPEG, WEBP, BMP, TIF, or TIFF images.")
            return redirect(url_for("index"))

        batch_id = uuid.uuid4().hex[:12]
        image_paths = []
        for index, uploaded in enumerate(uploads):
            item_run_id = f"{batch_id}_{index:03d}"
            run_input_dir = UPLOAD_DIR / item_run_id
            run_input_dir.mkdir(parents=True, exist_ok=True)
            filename = secure_filename(uploaded.filename)
            if not filename:
                filename = f"upload_{index}.png"
            image_path = run_input_dir / filename
            uploaded.save(image_path)
            image_paths.append(image_path)

        results = analyze_upload_batch(image_paths, batch_id)
        successful_results = [result for result in results if not result.get("error")]
        if not successful_results:
            for image_path in image_paths:
                shutil.rmtree(image_path.parent, ignore_errors=True)
            flash(f"Analysis failed: {results[0].get('error') if results else 'unknown error'}")
            return redirect(url_for("index"))

        return render_template("index.html", result=successful_results[0], results=results)

    return render_template("index.html", result=None, results=[])


@app.route("/uploads/<run_id>/<path:filename>")
def serve_upload_file(run_id: str, filename: str):
    return send_from_directory(UPLOAD_DIR / run_id, filename)


@app.route("/results/<run_id>/<path:filename>")
def serve_result_file(run_id: str, filename: str):
    return send_from_directory(RESULTS_DIR / run_id, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

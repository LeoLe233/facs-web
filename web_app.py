from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from facs_anime_analysis import (
    IMAGE_EXTENSIONS,
    emotion_columns,
)


BASE_DIR = Path(__file__).resolve().parent
WEB_RUNS_DIR = BASE_DIR / "web_runs"
UPLOAD_DIR = WEB_RUNS_DIR / "uploads"
RESULTS_DIR = WEB_RUNS_DIR / "results"
ALLOWED_EXTENSIONS = {extension.lstrip(".") for extension in IMAGE_EXTENSIONS}


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-facs-demo")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def static_result_url(run_id: str, relative_path: str) -> str:
    return url_for("serve_result_file", run_id=run_id, filename=relative_path)


def row_records(df: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    if df.empty:
        return []
    clean = df.head(limit).copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def analyze_upload(image_path: Path, run_id: str) -> dict[str, object]:
    run_input_dir = UPLOAD_DIR / run_id
    run_output_dir = RESULTS_DIR / run_id
    run_input_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
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
    completed = subprocess.run(command, cwd=BASE_DIR, text=True, capture_output=True)
    if completed.returncode != 0 and not (run_output_dir / "au_results.csv").exists():
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(details[-1200:])

    results = read_csv_if_exists(run_output_dir / "au_results.csv")
    detection_summary = read_csv_if_exists(run_output_dir / "detection_summary.csv")
    au_summary = read_csv_if_exists(run_output_dir / "au_summary.csv")
    emotion_predictions = read_csv_if_exists(run_output_dir / "emotion_predictions.csv")
    emotion_summary = read_csv_if_exists(run_output_dir / "emotion_summary.csv")

    emotion_cols = emotion_columns(results)
    detections = results.copy()
    detections["face_index"] = detections.groupby("image_path").cumcount() if "image_path" in detections else 0
    detections = detections.where(pd.notna(detections), None)

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
                aus.append({"name": col, "score": float(value)})
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
        "original_url": url_for("serve_upload_file", run_id=run_id, filename=image_path.name),
        "overlay_urls": [static_result_url(run_id, f"au_overlays/{path.name}") for path in overlay_paths],
        "plot_urls": plot_paths,
        "detection_summary": row_records(detection_summary),
        "emotion_predictions": row_records(emotion_predictions),
        "emotion_summary": row_records(emotion_summary),
        "au_summary": row_records(au_summary),
        "face_cards": face_cards,
        "downloads": {
            "AU results": static_result_url(run_id, "au_results.csv"),
            "Detection summary": static_result_url(run_id, "detection_summary.csv"),
            "AU summary": static_result_url(run_id, "au_summary.csv") if (run_output_dir / "au_summary.csv").exists() else None,
            "Emotion predictions": static_result_url(run_id, "emotion_predictions.csv") if (run_output_dir / "emotion_predictions.csv").exists() else None,
            "Emotion summary": static_result_url(run_id, "emotion_summary.csv") if (run_output_dir / "emotion_summary.csv").exists() else None,
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            flash("Choose an image first.")
            return redirect(url_for("index"))
        if not allowed_file(uploaded.filename):
            flash("Upload a PNG, JPG, JPEG, WEBP, BMP, TIF, or TIFF image.")
            return redirect(url_for("index"))

        run_id = uuid.uuid4().hex[:12]
        run_input_dir = UPLOAD_DIR / run_id
        run_input_dir.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(uploaded.filename)
        image_path = run_input_dir / filename
        uploaded.save(image_path)

        try:
            result = analyze_upload(image_path, run_id)
        except Exception as exc:
            shutil.rmtree(UPLOAD_DIR / run_id, ignore_errors=True)
            shutil.rmtree(RESULTS_DIR / run_id, ignore_errors=True)
            flash(f"Analysis failed: {exc}")
            return redirect(url_for("index"))

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)


@app.route("/uploads/<run_id>/<path:filename>")
def serve_upload_file(run_id: str, filename: str):
    return send_from_directory(UPLOAD_DIR / run_id, filename)


@app.route("/results/<run_id>/<path:filename>")
def serve_result_file(run_id: str, filename: str):
    return send_from_directory(RESULTS_DIR / run_id, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

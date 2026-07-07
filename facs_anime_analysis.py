from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AU_PREFIXES = ("AU", "au")
EMOTION_COLUMNS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_dir: Path
    backend: str
    metadata_csv: Path | None
    group_by: str | None
    openface_bin: str
    au_overlay_threshold: float
    au_overlay_top_n: int


AU_DESCRIPTIONS = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU28": "Lip Suck",
    "AU43": "Eyes Closed",
}


AU_REGION_GROUPS = {
    "AU01": ("inner_brow",),
    "AU02": ("outer_brow",),
    "AU04": ("inner_brow", "nose_bridge"),
    "AU05": ("eyes",),
    "AU06": ("cheeks",),
    "AU07": ("eyes",),
    "AU09": ("nose",),
    "AU10": ("upper_lip", "nose_base"),
    "AU11": ("nasolabial",),
    "AU12": ("mouth_corners", "cheeks"),
    "AU14": ("mouth_corners",),
    "AU15": ("lower_mouth",),
    "AU17": ("chin",),
    "AU20": ("mouth_wide",),
    "AU23": ("mouth",),
    "AU24": ("mouth",),
    "AU25": ("inner_mouth",),
    "AU26": ("jaw_mouth",),
    "AU28": ("mouth",),
    "AU43": ("eyes",),
}


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_image_key(path_value: str | Path) -> str:
    return Path(path_value).name.lower()


def au_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith(AU_PREFIXES) and (col.endswith("_r") or col.endswith("_c") or col[2:4].isdigit())
    ]


def normalize_au_code(value: str) -> str:
    upper = value.upper()
    if upper.startswith("AU") and len(upper) >= 4 and upper[2:4].isdigit():
        return upper[:4]
    return upper


def describe_au(value: str) -> str:
    return AU_DESCRIPTIONS.get(normalize_au_code(value), "Unknown AU description")


def au_reference_frame(cols: Iterable[str] | None = None) -> pd.DataFrame:
    codes = sorted({normalize_au_code(col) for col in cols} if cols else set(AU_DESCRIPTIONS))
    rows = []
    for code in codes:
        rows.append(
            {
                "au": code,
                "description": describe_au(code),
                "overlay_regions": ", ".join(AU_REGION_GROUPS.get(code, ())),
            }
        )
    return pd.DataFrame(rows)


def emotion_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in EMOTION_COLUMNS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def load_metadata(metadata_csv: Path | None) -> pd.DataFrame | None:
    if metadata_csv is None:
        return None

    metadata = pd.read_csv(metadata_csv)
    if "image_path" not in metadata.columns and "filename" not in metadata.columns:
        raise ValueError("Metadata CSV must include either an image_path or filename column.")

    key_column = "image_path" if "image_path" in metadata.columns else "filename"
    metadata = metadata.copy()
    metadata["image_key"] = metadata[key_column].map(normalize_image_key)
    return metadata


def attach_metadata(results: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
    if metadata is None:
        return results

    results = results.copy()
    results["image_key"] = results["image_path"].map(normalize_image_key)
    metadata_columns = [col for col in metadata.columns if col not in {"image_path", "filename"}]
    return results.merge(metadata[metadata_columns], on="image_key", how="left")


def run_openface(config: AnalysisConfig, images: list[Path]) -> pd.DataFrame:
    requested_binary = os.environ.get("OPENFACE_BIN", config.openface_bin)
    binary = shutil.which(requested_binary) or requested_binary
    if not Path(binary).exists() and shutil.which(binary) is None:
        raise RuntimeError(
            f"OpenFace binary not found: {requested_binary}.\n\n"
            "OpenFace is a separate compiled application, not a Python package. "
            "After installing it, run this script with one of these forms:\n"
            "  python facs_anime_analysis.py --backend openface --openface-bin /path/to/FeatureExtraction\n"
            "  OPENFACE_BIN=/path/to/FeatureExtraction python facs_anime_analysis.py --backend openface\n\n"
            "If you do not have OpenFace installed yet, use --backend pyfeat or install/build OpenFace first."
        )

    raw_dir = config.output_dir / "openface_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        binary,
        "-fdir",
        str(config.input_dir),
        "-out_dir",
        str(raw_dir),
        "-aus",
        "-q",
    ]
    subprocess.run(command, check=True)

    rows: list[pd.DataFrame] = []
    by_stem = {image.stem: image for image in images}

    for csv_path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue

        image_path = by_stem.get(csv_path.stem)
        if image_path is None:
            matches = [path for path in images if path.stem == csv_path.stem]
            image_path = matches[0] if matches else config.input_dir / csv_path.stem

        frame.insert(0, "image_path", str(image_path))
        frame.insert(1, "filename", image_path.name)
        frame.insert(2, "backend", "openface")
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["image_path", "filename", "backend", "success"])

    results = pd.concat(rows, ignore_index=True)
    if "success" not in results.columns:
        results["success"] = True
    return results


def run_pyfeat(config: AnalysisConfig, images: list[Path]) -> pd.DataFrame:
    if sys.version_info >= (3, 12):
        raise RuntimeError(
            "The Py-Feat backend requires Python 3.10 or 3.11 for this project. "
            f"You are running Python {sys.version_info.major}.{sys.version_info.minor}. "
            "On Windows, recreate the environment with `py -3.11 -m venv .venv311`, "
            "activate it, then run `python -m pip install -r requirements.txt`."
        )

    try:
        from feat.detector import Detectorv1
    except ImportError as exc:
        raise RuntimeError(
            "Py-Feat is not installed in this Python environment. "
            "Install it with `pip install py-feat`, or use --backend openface."
        ) from exc
    except Exception as exc:
        if "torchcodec" in str(exc).lower() or "libtorchcodec" in str(exc).lower():
            raise RuntimeError(
                "Py-Feat loaded, but TorchCodec could not load its Windows DLLs. "
                "Use Python 3.11, reinstall the requirements inside that environment, "
                "and install a full-shared FFmpeg build on Windows so its DLLs are on PATH. "
                "A clean reset is usually fastest: `rmdir /s /q .venv311`, "
                "`py -3.11 -m venv .venv311`, `.venv311\\Scripts\\activate`, "
                "`python -m pip install --upgrade pip`, then "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        raise

    detector = Detectorv1(device="cpu")
    rows: list[pd.DataFrame] = []

    for image_path in tqdm(images, desc="Running Py-Feat"):
        try:
            detected = detector.detect(str(image_path), progress_bar=False)
            frame = pd.DataFrame(detected)
            if frame.empty:
                frame = pd.DataFrame([{"success": False}])
            else:
                face_cols = [col for col in ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight") if col in frame]
                frame["success"] = ~frame[face_cols].isna().all(axis=1) if face_cols else True
        except Exception as exc:  # The exception text is useful for failure analysis.
            frame = pd.DataFrame([{"success": False, "error": str(exc)}])

        frame.insert(0, "image_path", str(image_path))
        frame.insert(1, "filename", image_path.name)
        frame.insert(2, "backend", "pyfeat")
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def summarize_detection(results: pd.DataFrame, group_by: str | None) -> pd.DataFrame:
    results = results.copy()
    results["success_bool"] = results["success"].astype(str).str.lower().isin({"1", "true", "yes"})

    grouping = [group_by] if group_by and group_by in results.columns else []
    summary = (
        results.groupby(grouping, dropna=False)["success_bool"]
        .agg(images="count", detected="sum", detection_rate="mean")
        .reset_index()
        if grouping
        else pd.DataFrame(
            [
                {
                    "images": len(results),
                    "detected": int(results["success_bool"].sum()),
                    "detection_rate": float(results["success_bool"].mean()) if len(results) else 0.0,
                }
            ]
        )
    )
    return summary


def summarize_aus(results: pd.DataFrame, group_by: str | None) -> pd.DataFrame:
    cols = [col for col in au_columns(results) if pd.api.types.is_numeric_dtype(results[col])]
    if not cols:
        return pd.DataFrame()

    success_mask = results["success"].astype(str).str.lower().isin({"1", "true", "yes"})
    detected = results.loc[success_mask].copy()

    if group_by and group_by in detected.columns:
        return detected.groupby(group_by, dropna=False)[cols].mean().reset_index()

    return detected[cols].mean().to_frame(name="mean").reset_index(names="au")


def summarize_emotions(results: pd.DataFrame, group_by: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = emotion_columns(results)
    if not cols:
        return pd.DataFrame(), pd.DataFrame()

    success_mask = results["success"].astype(str).str.lower().isin({"1", "true", "yes"})
    detected = results.loc[success_mask].copy()
    if detected.empty:
        return pd.DataFrame(), pd.DataFrame()

    detected["face_index"] = detected.groupby("image_path").cumcount()
    top_emotion = detected[cols].idxmax(axis=1)
    top_score = detected[cols].max(axis=1)

    ranked_rows = []
    for _, row in detected.iterrows():
        ranked = row[cols].sort_values(ascending=False)
        record = {
            "image_path": row.get("image_path"),
            "filename": row.get("filename"),
            "face_index": int(row["face_index"]),
            "top_emotion": ranked.index[0],
            "top_emotion_score": float(ranked.iloc[0]),
        }
        for rank, (emotion, score) in enumerate(ranked.head(3).items(), start=1):
            record[f"emotion_{rank}"] = emotion
            record[f"emotion_{rank}_score"] = float(score)
        ranked_rows.append(record)

    per_face = pd.DataFrame(ranked_rows)

    if group_by and group_by in detected.columns:
        summary = detected.groupby(group_by, dropna=False)[cols].mean().reset_index()
        summary["top_emotion"] = summary[cols].idxmax(axis=1)
        summary["top_emotion_score"] = summary[cols].max(axis=1)
    else:
        means = detected[cols].mean()
        summary = means.to_frame(name="mean").reset_index(names="emotion")
        summary = summary.sort_values("mean", ascending=False)

    return per_face, summary


def save_plots(
    results: pd.DataFrame,
    detection_summary: pd.DataFrame,
    au_summary: pd.DataFrame,
    emotion_summary: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4))
    if config.group_by and config.group_by in detection_summary.columns:
        sns.barplot(data=detection_summary, x=config.group_by, y="detection_rate")
        plt.xticks(rotation=30, ha="right")
    else:
        sns.barplot(data=detection_summary.assign(group="all"), x="group", y="detection_rate")
    plt.ylim(0, 1)
    plt.title("Face Detection Rate")
    plt.tight_layout()
    plt.savefig(plots_dir / "detection_rate.png", dpi=180)
    plt.close()

    if not au_summary.empty:
        if config.group_by and config.group_by in au_summary.columns:
            heatmap_data = au_summary.set_index(config.group_by)
        else:
            heatmap_data = au_summary.set_index("au").T

        numeric_heatmap = heatmap_data.select_dtypes("number")
        if not numeric_heatmap.empty:
            plt.figure(figsize=(max(8, numeric_heatmap.shape[1] * 0.55), max(3, numeric_heatmap.shape[0] * 0.55)))
            sns.heatmap(numeric_heatmap, cmap="viridis", annot=False)
            plt.title("Mean AU Output")
            plt.tight_layout()
            plt.savefig(plots_dir / "mean_au_heatmap.png", dpi=180)
            plt.close()

    if emotion_summary.empty:
        return

    emotion_cols = [col for col in EMOTION_COLUMNS if col in emotion_summary.columns]
    if emotion_cols and config.group_by and config.group_by in emotion_summary.columns:
        emotion_heatmap = emotion_summary.set_index(config.group_by)[emotion_cols]
    elif {"emotion", "mean"}.issubset(emotion_summary.columns):
        emotion_heatmap = emotion_summary.set_index("emotion")[["mean"]].T
    else:
        return

    plt.figure(figsize=(max(7, emotion_heatmap.shape[1] * 0.8), max(3, emotion_heatmap.shape[0] * 0.7)))
    sns.heatmap(emotion_heatmap, cmap="magma", vmin=0, vmax=1, annot=True, fmt=".2f")
    plt.title("Mean Emotion Probability")
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_emotion_heatmap.png", dpi=180)
    plt.close()


def landmark_points(row: pd.Series) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for idx in range(68):
        x_col = f"x_{idx}"
        y_col = f"y_{idx}"
        if x_col not in row or y_col not in row:
            continue
        x = row[x_col]
        y = row[y_col]
        if pd.isna(x) or pd.isna(y):
            continue
        points[idx] = (float(x), float(y))
    return points


def expanded_polygon(points: list[tuple[float, float]], scale: float = 1.25) -> list[tuple[float, float]]:
    if not points:
        return []

    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in points]


def points_for_indices(points: dict[int, tuple[float, float]], indices: Iterable[int]) -> list[tuple[float, float]]:
    return [points[idx] for idx in indices if idx in points]


def face_region_polygons(row: pd.Series) -> dict[str, list[list[tuple[float, float]]]]:
    points = landmark_points(row)
    if not points:
        return {}

    polygons = {
        "inner_brow": [
            expanded_polygon(points_for_indices(points, [19, 20, 23, 24, 27]), 1.55),
        ],
        "outer_brow": [
            expanded_polygon(points_for_indices(points, [17, 18, 19, 36, 39]), 1.4),
            expanded_polygon(points_for_indices(points, [24, 25, 26, 42, 45]), 1.4),
        ],
        "nose_bridge": [
            expanded_polygon(points_for_indices(points, [21, 22, 27, 28, 29]), 1.6),
        ],
        "eyes": [
            expanded_polygon(points_for_indices(points, [36, 37, 38, 39, 40, 41]), 1.35),
            expanded_polygon(points_for_indices(points, [42, 43, 44, 45, 46, 47]), 1.35),
        ],
        "cheeks": [
            expanded_polygon(points_for_indices(points, [1, 2, 31, 48, 49, 50]), 1.18),
            expanded_polygon(points_for_indices(points, [14, 15, 35, 52, 53, 54]), 1.18),
        ],
        "nose": [
            expanded_polygon(points_for_indices(points, [27, 28, 29, 30, 31, 32, 33, 34, 35]), 1.2),
        ],
        "nose_base": [
            expanded_polygon(points_for_indices(points, [31, 32, 33, 34, 35, 50, 51, 52]), 1.15),
        ],
        "upper_lip": [
            expanded_polygon(points_for_indices(points, [48, 49, 50, 51, 52, 53, 54, 61, 62, 63]), 1.2),
        ],
        "nasolabial": [
            expanded_polygon(points_for_indices(points, [31, 32, 48, 49, 50, 3]), 1.18),
            expanded_polygon(points_for_indices(points, [35, 34, 54, 53, 52, 13]), 1.18),
        ],
        "mouth_corners": [
            expanded_polygon(points_for_indices(points, [48, 49, 59, 60, 3, 31]), 1.25),
            expanded_polygon(points_for_indices(points, [54, 53, 55, 64, 13, 35]), 1.25),
        ],
        "mouth": [
            expanded_polygon(points_for_indices(points, range(48, 68)), 1.18),
        ],
        "mouth_wide": [
            expanded_polygon(points_for_indices(points, [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]), 1.35),
        ],
        "inner_mouth": [
            expanded_polygon(points_for_indices(points, range(60, 68)), 1.35),
        ],
        "lower_mouth": [
            expanded_polygon(points_for_indices(points, [48, 54, 55, 56, 57, 58, 59, 6, 8, 10]), 1.15),
        ],
        "chin": [
            expanded_polygon(points_for_indices(points, [6, 7, 8, 9, 10, 56, 57, 58]), 1.18),
        ],
        "jaw_mouth": [
            expanded_polygon(points_for_indices(points, [48, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9, 10]), 1.18),
        ],
    }

    return {
        name: [polygon for polygon in region_polygons if len(polygon) >= 3]
        for name, region_polygons in polygons.items()
    }


def active_aus(row: pd.Series, cols: list[str], threshold: float, top_n: int) -> list[tuple[str, float]]:
    scored = []
    for col in cols:
        value = row[col]
        if pd.isna(value):
            continue
        au = normalize_au_code(col)
        scored.append((au, float(value)))

    active = [(au, score) for au, score in scored if score >= threshold]
    if not active and top_n > 0:
        active = sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]

    return sorted(active, key=lambda item: item[1], reverse=True)


def save_au_overlays(results: pd.DataFrame, config: AnalysisConfig) -> None:
    cols = [col for col in au_columns(results) if pd.api.types.is_numeric_dtype(results[col])]
    if not cols:
        return

    overlay_dir = config.output_dir / "au_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("turbo")

    for row_index, row in results.iterrows():
        if str(row.get("success", "")).lower() not in {"1", "true", "yes"}:
            continue

        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue

        regions = face_region_polygons(row)
        if not regions:
            continue

        active = active_aus(row, cols, config.au_overlay_threshold, config.au_overlay_top_n)
        if not active:
            continue

        image = Image.open(image_path).convert("RGB")
        fig_width = 8
        fig_height = max(5, fig_width * image.height / image.width)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(image)
        ax.axis("off")

        emotions = emotion_columns(pd.DataFrame([row]))
        emotion_line = ""
        if emotions:
            ranked_emotions = row[emotions].dropna().sort_values(ascending=False)
            if not ranked_emotions.empty:
                emotion_line = f"Emotion: {ranked_emotions.index[0]} {ranked_emotions.iloc[0]:.2f}\n"

        face_box_cols = ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight")
        if all(col in row and not pd.isna(row[col]) for col in face_box_cols):
            ax.add_patch(
                Rectangle(
                    (float(row["FaceRectX"]), float(row["FaceRectY"])),
                    float(row["FaceRectWidth"]),
                    float(row["FaceRectHeight"]),
                    fill=False,
                    edgecolor="white",
                    linewidth=3.0,
                    linestyle="--",
                    alpha=0.95,
                )
            )

        points = landmark_points(row)
        if points:
            xs = [point[0] for point in points.values()]
            ys = [point[1] for point in points.values()]
            ax.scatter(xs, ys, s=12, c="white", edgecolors="black", linewidths=0.45, alpha=0.9)

        label_lines = []
        for au_index, (au, score) in enumerate(active):
            color = cmap(au_index / max(1, len(active) - 1))
            label_lines.append(f"{au} {score:.2f} - {describe_au(au)}")

            for region_name in AU_REGION_GROUPS.get(au, ()):
                for polygon in regions.get(region_name, []):
                    patch = Polygon(
                        polygon,
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=3.0,
                        alpha=0.5,
                    )
                    ax.add_patch(patch)

        ax.text(
            0.015,
            0.985,
            emotion_line + "\n".join(label_lines[:10]),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.68, "edgecolor": "none", "pad": 6},
        )

        output_name = f"{image_path.stem}_face{row_index}_au_overlay.png"
        fig.tight_layout(pad=0)
        fig.savefig(overlay_dir / output_name, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def write_template_metadata(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        [
            {
                "filename": "example.png",
                "style": "anime",
                "expected_expression": "happy",
                "source": "manual",
                "notes": "replace with your own filenames",
            }
        ]
    )
    template.to_csv(path, index=False)


def parse_args(argv: Iterable[str]) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Run FACS/AU analysis on anime, comic, or human-face images.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/images"), help="Folder containing input images.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Folder for CSVs and plots.")
    parser.add_argument("--backend", choices=("pyfeat", "openface"), default="pyfeat", help="AU detector backend.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/metadata.csv"), help="Optional image metadata CSV.")
    parser.add_argument("--group-by", default="style", help="Metadata column for grouped summaries, e.g. style or expected_expression.")
    parser.add_argument(
        "--au-overlay-threshold",
        type=float,
        default=0.5,
        help="Minimum AU score to draw on image overlays.",
    )
    parser.add_argument(
        "--au-overlay-top-n",
        type=int,
        default=5,
        help="If no AU reaches the threshold, draw this many highest-scoring AUs.",
    )
    parser.add_argument(
        "--openface-bin",
        default="FeatureExtraction",
        help="OpenFace FeatureExtraction binary path. Can also be set with OPENFACE_BIN.",
    )
    args = parser.parse_args(list(argv))

    return AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        metadata_csv=args.metadata_csv if args.metadata_csv.exists() else None,
        group_by=args.group_by,
        openface_bin=args.openface_bin,
        au_overlay_threshold=args.au_overlay_threshold,
        au_overlay_top_n=args.au_overlay_top_n,
    )


def main(argv: Iterable[str] | None = None) -> int:
    config = parse_args(sys.argv[1:] if argv is None else argv)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_template_metadata(Path("data/metadata.csv"))

    images = iter_images(config.input_dir)
    if not images:
        print(f"No images found in {config.input_dir}. Add files with extensions: {sorted(IMAGE_EXTENSIONS)}")
        return 1

    metadata = load_metadata(config.metadata_csv)

    if config.backend == "openface":
        results = run_openface(config, images)
    else:
        results = run_pyfeat(config, images)

    results = attach_metadata(results, metadata)
    detection_summary = summarize_detection(results, config.group_by)
    au_summary = summarize_aus(results, config.group_by)
    emotion_predictions, emotion_summary = summarize_emotions(results, config.group_by)

    results.to_csv(config.output_dir / "au_results.csv", index=False)
    detection_summary.to_csv(config.output_dir / "detection_summary.csv", index=False)
    au_reference_frame(au_columns(results)).to_csv(config.output_dir / "au_reference.csv", index=False)
    if not au_summary.empty:
        au_summary.to_csv(config.output_dir / "au_summary.csv", index=False)
    if not emotion_predictions.empty:
        emotion_predictions.to_csv(config.output_dir / "emotion_predictions.csv", index=False)
    if not emotion_summary.empty:
        emotion_summary.to_csv(config.output_dir / "emotion_summary.csv", index=False)

    save_plots(results, detection_summary, au_summary, emotion_summary, config)
    save_au_overlays(results, config)

    print(f"Wrote {config.output_dir / 'au_results.csv'}")
    print(f"Wrote {config.output_dir / 'au_reference.csv'}")
    print(f"Wrote {config.output_dir / 'detection_summary.csv'}")
    if not au_summary.empty:
        print(f"Wrote {config.output_dir / 'au_summary.csv'}")
    if not emotion_predictions.empty:
        print(f"Wrote {config.output_dir / 'emotion_predictions.csv'}")
    if not emotion_summary.empty:
        print(f"Wrote {config.output_dir / 'emotion_summary.csv'}")
    print(f"Wrote AU overlays to {config.output_dir / 'au_overlays'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

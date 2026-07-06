# FACS/AU Testing Tool

This project is a small research harness for testing common facial Action Unit (OpenFace/py-feat) tools.

It records two things that matter for your question:

- Whether the tool detects a face at all.
- Which AU scores/classes it returns when detection succeeds.

Runs as a website based on Flask on port 5001.


## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`py-feat` is included only for Python versions below 3.12 because many AU model stacks lag behind the newest Python releases. If you are on Python 3.12 and Py-Feat does not install, create the environment with Python 3.10 or 3.11, or use the OpenFace backend.

## Add Images

Put images into:

```text
data/images/
```

Then edit `data/metadata.csv` so each row names a file:

```csv
filename,style,expected_expression,source,notes
happy_01.png,anime,happy,manual,
smile_panel.jpg,comic,happy,manual,
human_smile.jpg,human,happy,comparison,
```

## Run With Py-Feat

```bash
python facs_anime_analysis.py --backend pyfeat --input-dir data/images --output-dir results --group-by style
```

## Run With OpenFace

Install OpenFace separately, then run:

```bash
python facs_anime_analysis.py --backend openface --input-dir data/images --output-dir results --group-by style
```

If the binary is not on your PATH:

```bash
python facs_anime_analysis.py --backend openface --openface-bin /path/to/FeatureExtraction
```

## Outputs

- `results/au_results.csv`: one row per detected frame/image, including AU columns when available.
- `results/au_reference.csv`: AU code, FACS action description, and approximate overlay region names.
- `results/detection_summary.csv`: detection count and detection rate overall or by metadata group.
- `results/au_summary.csv`: mean AU values for detected images.
- `results/emotion_predictions.csv`: top 3 possible emotions for each detected face.
- `results/emotion_summary.csv`: mean emotion probabilities overall or by metadata group.
- `results/plots/detection_rate.png`: face-detection success rate.
- `results/plots/mean_au_heatmap.png`: average AU output.
- `results/plots/mean_emotion_heatmap.png`: average emotion probabilities.
- `results/au_overlays/`: processed images with detected AU face regions highlighted.

## AU Region Overlays

Each run creates annotated copies of detected images. Highlighted regions are based on the 68 facial landmarks returned by the detector, so they should be treated as approximate region visualizations rather than exact FACS muscle boundaries.

By default, the script draws AUs with scores of `0.5` or higher. If no AU reaches that threshold, it draws the top 5 AUs so every detected face has a useful diagnostic image.

```bash
python facs_anime_analysis.py --backend pyfeat --input-dir data/images --output-dir results --au-overlay-threshold 0.4
```

## Suggested Study Design

Use a matched set with the same intended expressions across styles. For example, collect 20 anime, 20 comic, and 20 human-reference images for each expression category. The most important result is not just the AU number, but where the detector fails completely or returns inconsistent AU patterns for clearly labeled expressions.

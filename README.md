# FACS/AU Testing Tool

This project is a small research harness for testing common facial Action Unit (OpenFace/py-feat) tools.

It records two things that matter for your question:

- Whether the tool detects a face at all.
- Which AU scores/classes it returns when detection succeeds.

Runs as a website based on Flask on port 5001.


## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeoLe233/facs-web.git
   cd facs-web
   ```

2. **Create and activate a virtual environment:**

   **On macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **On Windows (Command Prompt):**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **On Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Note:** `py-feat` is included only for Python versions below 3.12 because many AU model stacks lag behind the newest Python releases. If you are on Python 3.12 and Py-Feat does not install, create the environment with Python 3.10 or 3.11, or use the OpenFace backend.

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

## AU Region Overlays

Each run creates annotated copies of detected images. Highlighted regions are based on the 68 facial landmarks returned by the detector, so they should be treated as approximate region visualizations rather than exact FACS muscle boundaries.

By default, the script draws AUs with scores of `0.5` or higher. If no AU reaches that threshold, it draws the top 5 AUs so every detected face has a useful diagnostic image.

```bash
python facs_anime_analysis.py --backend pyfeat --input-dir data/images --output-dir results --au-overlay-threshold 0.4
```

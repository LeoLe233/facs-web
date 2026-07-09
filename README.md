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

2. **Create and activate a Python 3.12 virtual environment:**

   **On macOS/Linux:**
   ```bash
   python3.12 -m venv .venv312
   source .venv312/bin/activate
   ```

   **On Windows (Command Prompt):**
   ```cmd
   py -3.12 -m venv .venv312
   .venv312\Scripts\activate
   ```

   **On Windows (PowerShell):**
   ```powershell
   py -3.12 -m venv .venv312
   .venv312\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Note:** Py-Feat 2.x requires Python 3.11 or newer. Python 3.12 is recommended on Windows when using AMD ROCm PyTorch.

## AMD GPU / ROCm PyTorch

The Py-Feat backend defaults to `--pyfeat-device cuda`. This is also the correct option for AMD GPUs when PyTorch is installed with ROCm support, because ROCm PyTorch exposes AMD GPUs through `torch.cuda`.

On Windows, follow AMD's ROCm 7.2.1 PyTorch wheel instructions for Python 3.12. After installing ROCm PyTorch, verify it:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

This project was verified with `torch==2.9.1+rocm7.2.1` detecting `AMD Radeon(TM) 8060S Graphics`.

To intentionally run without GPU acceleration:

```bash
python facs_anime_analysis.py --backend pyfeat --pyfeat-device cpu
```

## Windows TorchCodec Fix

If Windows shows an error like `Could not find module ... torchcodec\libtorchcodec_core4.dll`, first check FFmpeg. TorchCodec needs FFmpeg DLLs, not just `ffmpeg.exe`, and the Gyan `essentials_build` / static build does not provide the shared DLLs TorchCodec loads.

Install a Windows FFmpeg **full-shared** build and make sure its `bin` folder is on `PATH`. You can also point the analyzer directly at a DLL folder:

```cmd
set FACS_FFMPEG_DLL_DIR=C:\path\to\ffmpeg\bin
python -c "from facs_anime_analysis import configure_windows_ffmpeg_dlls, install_image_only_torchcodec_stub_if_needed; configure_windows_ffmpeg_dlls(); install_image_only_torchcodec_stub_if_needed(); from feat.detector import Detectorv1; print('py-feat ok')"
python web_app.py
```

The folder should contain files like `avcodec-*.dll`, `avformat-*.dll`, and `avutil-*.dll`.

With AMD's Windows ROCm PyTorch wheels, TorchCodec can still be binary-incompatible even when FFmpeg DLLs are present. This project only analyzes images, so the analyzer installs an image-only TorchCodec fallback during Py-Feat import. Video decoding through Py-Feat is intentionally unavailable in that fallback.

From the repository folder in Command Prompt:

```cmd
rmdir /s /q .venv312
py -3.12 -m venv .venv312
.venv312\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable); from facs_anime_analysis import install_image_only_torchcodec_stub_if_needed; install_image_only_torchcodec_stub_if_needed(); from feat.detector import Detectorv1; print('py-feat ok')"
python web_app.py
```

If the app still uses the wrong Python, set `FACS_ANALYZER_PYTHON` to the virtual environment interpreter before starting Flask:

```cmd
set FACS_ANALYZER_PYTHON=%CD%\.venv312\Scripts\python.exe
python web_app.py
```

## Run With Py-Feat

```bash
python facs_anime_analysis.py --backend pyfeat --pyfeat-device cuda --input-dir data/images --output-dir results --group-by style
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

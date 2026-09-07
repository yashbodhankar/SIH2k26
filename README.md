# LunarMatch-AI

A functional prototype for SIH 2026 Problem Statement 26166: multi-modal, sun-angle and scale-invariant correspondence between lunar images.

## What is implemented

- Local Streamlit mission-control dashboard with source/reference upload and synthetic demo mode.
- Robust grayscale/intensity normalization, optional CLAHE, conservative denoising, gradient/orientation/edge representations, and configurable image pyramids.
- SIFT and ORB baselines with Lowe ratio filtering and mutual consistency.
- Structural SIFT mode that merges normalized-intensity and gradient-based matches
	for illumination and sensor appearance differences.
- Multi-scale feature matching and gradient region retrieval for full-disk versus
	cropped-terrain image pairs.
- Configurable confidence filtering and N x N spatial distribution control.
- Similarity, affine, and homography RANSAC estimation with model selection using inlier count and reprojection error.
- Registered image, overlay, match visualization, structural representation viewer, CSV and PNG export.
- Local template-based refinement of reliable inlier points.
- Metrics: inlier count/ratio, RMSE, mean reprojection error, spatial coverage, match density proxy, and clearly labeled Prototype Registration Confidence.
- Structural similarity score based on gradient correlation after registration.
- Automated synthetic translation registration test.

The optional `Advanced AI (fallback)` selection currently uses the SIFT structural baseline. This keeps the prototype offline and reproducible; a SuperPoint/LightGlue or LoFTR adapter can be added behind the matching interface when model weights and compatible runtime packages are available.

## Setup

### Windows PowerShell

```powershell
cd d:\SIH
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or newer is recommended. Python 3.13 may work if compatible wheels are available.

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Demo mode is enabled by default and clearly marked as synthetic validation data, not official ISRO evaluation data.

## Tests

```bash
pytest -q
```

## Architecture

`backend/preprocessing` owns validation and structural representations. `backend/matching` owns classical detectors and spatial balancing. `backend/geometry` owns robust transformation estimation. `backend/refinement` owns local refinement. `backend/evaluation` owns metrics. `backend/registration` composes these modules without coupling them to the UI.

## Scientific and operational notes

- Absolute brightness is not used as the only correspondence signal; normalized structural features drive the baseline.
- Confidence and registration values are computed from the current input pair. No benchmark values are hard-coded.
- Prototype Registration Confidence is an interpretable demo quality indicator, not an official ISRO metric.
- TIFF loading is delegated to OpenCV and may be limited by codec support. Large uploads should be resized or tiled in a production deployment.
- Processing is local by default. Uploads are passed to the in-memory pipeline and are not executed or persisted by the app.

## Future extensions

Add sensor-aware multi-channel handling, pretrained SuperPoint/LightGlue or LoFTR adapters, coarse-to-fine learned matching, phase-correlation sub-pixel refinement, GeoTIFF metadata preservation, experiment comparison runs, PDF reports, GPU batching, DEM-assisted orthorectification, uncertainty estimation, and official Chandrayaan benchmark evaluation.

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from backend.config import RegistrationConfig
from backend.evaluation.export import encode_png, matches_csv, metrics_json, result_json
from backend.evaluation.synthetic import create_synthetic_pair
from backend.registration.pipeline import register

st.set_page_config(page_title="LunarMatch-AI", page_icon="◐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root { --ink:#e8edf2; --muted:#8492a1; --line:#25323c; --cyan:#62d8d0; --amber:#f0b35c; }
html, body, [class*="css"] { font-family:'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, #17252d 0, #0b1116 45%, #080c10 100%); color:var(--ink); }
.block-container { max-width: 1440px; padding-top: 2rem; }
.mono { font-family:'DM Mono', monospace; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; font-size:.72rem; }
.hero { border-bottom:1px solid var(--line); padding-bottom:1.2rem; margin-bottom:1.4rem; }
.hero h1 { font-size:3rem; letter-spacing:-.04em; margin:.2rem 0; color:#f4f7f8; }
.hero p { color:var(--muted); margin:0; }
.panel { border:1px solid var(--line); background:rgba(14,22,28,.78); padding:1.1rem; min-height:220px; }
.metric { border-top:2px solid var(--cyan); padding-top:.6rem; }
.metric b { display:block; font-size:1.55rem; color:#f4f7f8; font-family:'DM Mono', monospace; }
.metric span { color:var(--muted); font-size:.75rem; text-transform:uppercase; }
div.stButton > button { background:#b8f1e9; color:#071013; border:0; border-radius:2px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


def decode(uploaded):
    return cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def image_bytes(image):
    return encode_png(image)


def make_demo(rotation=7.0, scale=0.88, noise_sigma=0.0):
    base = np.zeros((620, 900), np.uint8)
    rng = np.random.default_rng(26166)
    base[:] = rng.normal(70, 18, base.shape).clip(0, 255)
    for _ in range(34):
        x, y = int(rng.integers(30, 870)), int(rng.integers(30, 590)); r = int(rng.integers(8, 48))
        cv2.circle(base, (x, y), r, int(rng.integers(35, 130)), 2)
        cv2.circle(base, (x - r // 3, y - r // 3), max(2, r // 8), 190, -1)
    cv2.line(base, (80, 490), (760, 120), 150, 5)
    transform = cv2.getRotationMatrix2D((450, 310), rotation, scale); transform[:, 2] += [52, -26]
    moved = cv2.warpAffine(base, transform, (900, 620)); moved = cv2.convertScaleAbs(moved, alpha=1.18, beta=12)
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, moved.shape)
        moved = np.clip(moved.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR), cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


st.markdown('<div class="hero"><div class="mono">SIH 2026 / PS 26166 / LOCAL PROTOTYPE</div><h1>LUNARMATCH-AI</h1><p>Multi-modal lunar image correspondence and registration</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="mono">Mission controls</div>', unsafe_allow_html=True)
    sensor = st.selectbox("Sensor", ["OHRC", "TMC-2", "IIRS", "LRO NAC", "SELENE", "Unknown"])
    method = st.selectbox("Matching method", ["Structural SIFT", "SIFT", "ORB", "Advanced AI (fallback)"])
    transform = st.selectbox("Transformation", ["auto", "similarity", "affine", "homography"])
    grid_size = st.slider("Distribution grid", 4, 12, 8)
    max_per_cell = st.slider("Max matches per cell", 1, 20, 10)
    min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.35, 0.05)
    refine = st.checkbox("Attempt sub-pixel refinement", True)
    demo = st.checkbox("Use demo data", True)
    demo_rotation = st.slider("Demo rotation", -30.0, 30.0, 7.0, 1.0)
    demo_scale = st.slider("Demo scale", 0.5, 1.5, 0.88, 0.01)
    demo_noise = st.slider("Demo noise", 0.0, 25.0, 0.0, 1.0)
    st.caption("Demo data is synthetic validation imagery, not official ISRO evaluation data.")

left, right = st.columns(2)
with left:
    st.markdown('<div class="mono">Source / moving image</div>', unsafe_allow_html=True)
    source_upload = st.file_uploader("Upload source image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="source")
with right:
    st.markdown('<div class="mono">Reference / fixed image</div>', unsafe_allow_html=True)
    reference_upload = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="reference")

source = decode(source_upload) if source_upload else None
reference = decode(reference_upload) if reference_upload else None
if demo and (source is None or reference is None):
    source, reference = make_demo(demo_rotation, demo_scale, demo_noise)
    st.info("DEMO DATA · Synthetic validation pair · Not official evaluation data")

if source is not None and reference is not None:
    preview_left, preview_right = st.columns(2)
    with preview_left: st.image(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), caption="Source", use_container_width=True)
    with preview_right: st.image(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), caption="Reference", use_container_width=True)

if st.button("RUN REGISTRATION", use_container_width=True):
    if source is None or reference is None:
        st.error("Upload both images or enable Demo Data.")
    else:
        try:
            with st.spinner("Extracting structural features, matching, and estimating geometry..."):
                result = register(source, reference, RegistrationConfig(method=method, transform=transform, grid_size=grid_size, max_matches_per_cell=max_per_cell, min_confidence=min_confidence, refine=refine))
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"REGISTRATION FAILED · {exc}")

result = st.session_state.get("result")
if result:
    metrics = result["metrics"]
    st.markdown("---")
    st.markdown('<div class="mono">Registration result</div>', unsafe_allow_html=True)
    cards = st.columns(7)
    values = [("Model", result["model"].title()), ("Matches", metrics["candidate_matches"]), ("Inliers", metrics["inliers"]), ("Inlier ratio", f'{metrics["inlier_ratio"]:.1%}'), ("RMSE", f'{metrics["rmse_px"]:.2f} px' if metrics["rmse_px"] is not None else "n/a"), ("Coverage", f'{metrics["spatial_coverage"]:.1%}'), ("Structural similarity", f'{metrics["structural_similarity"]:.1%}')]
    for card, (label, value) in zip(cards, values):
        card.markdown(f'<div class="metric"><span>{label}</span><b>{value}</b></div>', unsafe_allow_html=True)
    st.caption(f'Prototype Registration Confidence: {metrics["prototype_registration_confidence"]:.1%} · Refined points: {result["refined_count"]} · Sensor: {sensor}')
    tab_registered, tab_overlay, tab_matches, tab_features = st.tabs(["Registered", "Overlay", "Correspondences", "Structure"])
    with tab_registered:
        st.image(cv2.cvtColor(result["registered"], cv2.COLOR_BGR2RGB), use_container_width=True)
    with tab_overlay:
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5)
        overlay = cv2.addWeighted(result["registered"], alpha, reference, 1 - alpha, 0)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
    with tab_matches:
        canvas_height = max(source.shape[0], reference.shape[0])
        canvas = np.zeros((canvas_height, source.shape[1] + reference.shape[1], 3), dtype=np.uint8)
        canvas[:source.shape[0], :source.shape[1]] = source
        canvas[:reference.shape[0], source.shape[1]:] = reference
        offset = source.shape[1]
        for match in result["matches"]:
            color = (80, 230, 180) if match.is_inlier else (90, 100, 230)
            p1 = (int(match.source_x), int(match.source_y)); p2 = (int(match.reference_x) + offset, int(match.reference_y))
            cv2.line(canvas, p1, p2, color, 1); cv2.circle(canvas, p1, 3, color, -1); cv2.circle(canvas, p2, 3, color, -1)
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="Green: RANSAC inliers · Red: rejected matches", use_container_width=True)
        st.download_button("Download matches.csv", matches_csv(result["matches"]), "matches.csv", "text/csv")
    with tab_features:
        feature_view = st.selectbox("Representation", ["Normalized", "Gradient", "Edges"])
        source_prepared = result["source_prepared"]
        image = {"Normalized": source_prepared.normalized, "Gradient": source_prepared.gradient, "Edges": source_prepared.edges}[feature_view]
        st.image(image, caption=feature_view, clamp=True, use_container_width=True)
    st.download_button("Download registered image", image_bytes(result["registered"]), "registered_image.png", "image/png")
    st.download_button("Download transformation.json", result_json(result), "transformation.json", "application/json")
    st.download_button("Download metrics.json", metrics_json(result["metrics"]), "metrics.json", "application/json")

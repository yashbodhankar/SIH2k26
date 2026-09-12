from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.config import RegistrationConfig
from backend.evaluation.export import encode_png, matches_csv, metrics_json, result_json
from backend.registration.pipeline import register

app = FastAPI(title="LunarMatch-AI API", version="0.1.0")


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a supported image")
    return image


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "LunarMatch-AI"}


@app.post("/api/register")
async def register_images(
    source_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    sensor: str = Form("Unknown"),
    method: str = Form("structural"),
    transformation: str = Form("auto"),
    min_confidence: float = Form(0.35),
) -> JSONResponse:
    try:
        source = decode_image(await source_image.read())
        reference = decode_image(await reference_image.read())
        result = register(
            source,
            reference,
            RegistrationConfig(
                method=method,
                transform=transformation,
                min_confidence=float(np.clip(min_confidence, 0.0, 1.0)),
            ),
        )
        payload = {
            "status": "success",
            "sensor": sensor,
            "transformation_model": result["model"],
            "metrics": result["metrics"],
            "refined_points": result["refined_count"],
            "transformation_json": result_json(result),
            "metrics_json": metrics_json(result["metrics"]),
            "matches_csv": matches_csv(result["matches"]),
            "registered_image_png": encode_png(result["registered"]).hex(),
        }
        return JSONResponse(payload)
    except (ValueError, cv2.error) as exc:
        return JSONResponse(status_code=422, content={"status": "failed", "reason": str(exc)})
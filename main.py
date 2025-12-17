from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from model import predict
from explainability import generate_gradcam_overlay, generate_intrinsic_maps

MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


app = FastAPI(
    title="Explainable AI Clinical Decision Support System",
    description="Hybrid XAI-based CDSS for Pneumonia Detection from Chest X-rays",
    version="1.0"
)

# Enable CORS (for React frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "AI backend running successfully"}



def ethical_notice():
    return {
        "disclaimer": (
            "This system is intended to support clinical decision-making only. "
            "It must not be used as a standalone diagnostic tool."
        ),
        "intended_use": (
            "For research and educational purposes in clinical decision support. "
            "Final diagnosis must be made by a qualified healthcare professional."
        ),
        "limitations": [
            "Model performance depends on image quality and dataset bias.",
            "Predictions may not generalize across different populations or imaging devices.",
            "The system has not been clinically validated for real-world deployment."
        ]
    }


@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    # ---------- File type validation ----------
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPG or PNG image."
        )

    # ---------- File size validation ----------
    image_bytes = await file.read()
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 5MB."
        )

    # ---------- Image decoding validation ----------
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    # ---------- Model prediction ----------
    prediction_result = predict(image)

    pred_class_index = 1 if prediction_result["label"] == "Pneumonia" else 0

    gradcam_overlay = generate_gradcam_overlay(
        image,
        prediction_result["image_tensor"],
        target_class=pred_class_index
    )

    # intrinsic_maps = generate_intrinsic_maps(
    #     prediction_result["image_tensor"]
    # )
    intrinsic_maps = None


    return {
        "prediction": prediction_result["label"],
        "confidence": prediction_result["confidence"],
        "explainability": {
            "gradcam_overlay": gradcam_overlay,
            "intrinsic_maps": intrinsic_maps
        },
        "ethics": ethical_notice()
    }


METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")

@app.get("/model/metrics")
def get_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"error": "Metrics not found. Compute metrics first."}
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
    return metrics
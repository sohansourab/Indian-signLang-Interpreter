import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ISL_Interpreter.keras"
LABELS_PATH = ROOT / "labels.json"
TEMPLATE_PATH = ROOT / "templates" / "index.html"

app = FastAPI(title="ISL Web Deployment")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found at {LABELS_PATH}.")

model = load_model(str(MODEL_PATH))
labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.reshape(normalized, (1, 128, 128, 1))


@app.get("/", response_class=HTMLResponse)
async def index():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="UI template not found")
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/predict")
async def predict(payload: dict):
    image_data = payload.get("image")
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing 'image' payload")

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image payload")

    processed = preprocess_frame(frame)
    prediction = model.predict(processed, verbose=0)[0]
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    return {
        "label": labels[class_index],
        "confidence": round(confidence, 4),
        "class_index": class_index,
        "all_scores": {label: round(float(score), 4) for label, score in zip(labels, prediction)},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)

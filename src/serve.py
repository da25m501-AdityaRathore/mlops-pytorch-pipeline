"""FastAPI serving app for the CIFAR-10 classifier.

Endpoints:
  GET  /health   -> 200 if a model checkpoint is loaded, 503 otherwise
  POST /predict  -> multipart/form-data upload, field name "image"

Config (env vars, all optional):
  CHECKPOINT_PATH   default: /app/checkpoints/classifier_v1.pt
  MODEL_ARCHITECTURE  default: resnet18 (overridden by checkpoint metadata if present)
  NUM_CLASSES       default: 10
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
DEFAULT_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
DEFAULT_NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)

model_state = {"model": None, "device": None}


def load_model() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT_PATH):
        logger.warning("No checkpoint found at %s; /predict will return 503", CHECKPOINT_PATH)
        model_state["model"] = None
        model_state["device"] = device
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    architecture = checkpoint.get("architecture", DEFAULT_ARCHITECTURE)
    num_classes = checkpoint.get("num_classes", DEFAULT_NUM_CLASSES)

    model = get_model(architecture=architecture, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_state["model"] = model
    model_state["device"] = device
    logger.info("Loaded checkpoint from %s (architecture=%s)", CHECKPOINT_PATH, architecture)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="mlops-pytorch-pipeline serving", lifespan=lifespan)


@app.get("/health")
def health():
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    device = model_state["device"]
    tensor = _inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_state["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx] if predicted_idx < len(CIFAR10_CLASSES) else predicted_idx,
        "predicted_index": predicted_idx,
        "probabilities": probs,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

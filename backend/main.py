from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.schemas import AnalyzeResponse
from backend.service import analyze_image
from models.zoo import list_models


app = FastAPI(title="XAI-Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models")
def get_models():
    return {"models": list_models()}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    model_name: str = Form("resnet18"),
    grid_size: int = Form(3),
    topk: int = Form(5),
    fill_mode: str = Form("constant"),
    constant_value: float = Form(0.0),
    noise_std: float = Form(0.15),
):
    if fill_mode not in {"constant", "mean", "noise"}:
        raise HTTPException(status_code=400, detail="Invalid fill_mode.")

    if grid_size < 2 or grid_size > 6:
        raise HTTPException(status_code=400, detail="grid_size must be between 2 and 6.")

    if topk < 1 or topk > 10:
        raise HTTPException(status_code=400, detail="topk must be between 1 and 10.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        return analyze_image(
            image=pil_image,
            model_name=model_name,
            grid_size=grid_size,
            fill_mode=fill_mode,
            constant_value=constant_value,
            noise_std=noise_std,
            topk=topk,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

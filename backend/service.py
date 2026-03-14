import base64
import io

import torch
from PIL import Image

from models.zoo import get_categories, get_preprocess, load_model
from modules.masker import ImageMasker
from modules.scorer import ImageScorer
from modules.visualizer import ImportanceVisualizer


_MODEL_CACHE = {}


def load_model_bundle(model_name, device):
    cache_key = (model_name, device)
    if cache_key not in _MODEL_CACHE:
        model, weights = load_model(model_name, pretrained=True, device=device)
        preprocess = get_preprocess(weights)
        categories = get_categories(weights)
        _MODEL_CACHE[cache_key] = (model, preprocess, categories)
    return _MODEL_CACHE[cache_key]


def predict_top1(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        prob = torch.softmax(output, dim=1)
        conf, cls = torch.max(prob, dim=1)
    return int(cls.item()), float(conf.item())


def pil_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_image(
    image,
    model_name,
    grid_size,
    fill_mode,
    constant_value,
    noise_std,
    topk,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, categories = load_model_bundle(model_name, device)

    analysis_image = image.convert("RGB").resize((224, 224))
    image_tensor = preprocess(analysis_image).to(device)

    original_class, original_conf = predict_top1(model, image_tensor)

    masker = ImageMasker(
        fill_mode=fill_mode,
        constant_value=constant_value,
        noise_std=noise_std,
    )
    masked_results = masker.generate_all(image_tensor, grid_size=grid_size)

    scorer = ImageScorer()
    scores = scorer.score(model, image_tensor, masked_results)
    ranked_scores = sorted(scores, key=lambda item: item["importance"], reverse=True)

    visualizer = ImportanceVisualizer()
    overlay = visualizer.render_overlay(analysis_image, ranked_scores, topk=topk)

    class_name = str(original_class)
    if categories and 0 <= original_class < len(categories):
        class_name = categories[original_class]

    top_regions = []
    for rank, item in enumerate(ranked_scores[:topk], start=1):
        y1, y2, x1, x2 = item["bbox"]
        top_regions.append(
            {
                "rank": rank,
                "bbox": {"y1": y1, "y2": y2, "x1": x1, "x2": x2},
                "importance": item["importance"],
            }
        )

    return {
        "device": device,
        "class_index": original_class,
        "class_name": class_name,
        "confidence": original_conf,
        "top_regions": top_regions,
        "overlay_image": pil_image_to_base64(overlay),
        "analysis_image": pil_image_to_base64(analysis_image),
    }

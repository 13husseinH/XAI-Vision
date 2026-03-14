import base64
import io

import torch
import torchvision.transforms.functional as TF

from models.zoo import get_categories, get_preprocess, load_model
from modules.masker import ImageMasker
from modules.scorer import ImageScorer
from modules.visualizer import ImportanceVisualizer


_MODEL_CACHE = {}
_IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
_IMAGE_NET_STD = (0.229, 0.224, 0.225)


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


def tensor_to_display_image(image_tensor):
    tensor = image_tensor.detach().cpu().clone()
    mean = torch.tensor(_IMAGE_NET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGE_NET_STD).view(3, 1, 1)
    tensor = (tensor * std + mean).clamp(0.0, 1.0)
    return TF.to_pil_image(tensor)


def build_region_payload(scores):
    payload = []
    for rank, item in enumerate(scores, start=1):
        y1, y2, x1, x2 = item["bbox"]
        payload.append(
            {
                "rank": rank,
                "bbox": {"y1": y1, "y2": y2, "x1": x1, "x2": x2},
                "importance": item["importance"],
                "original_confidence": item["original_confidence"],
                "masked_confidence": item["masked_confidence"],
                "mask_type": item["mask_type"],
                "label": item["label"],
            }
        )
    return payload


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

    image_tensor = preprocess(image.convert("RGB")).to(device)
    analysis_image = tensor_to_display_image(image_tensor)

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
    heatmap_overlay = visualizer.render_heatmap_overlay(analysis_image, ranked_scores)
    box_overlay = visualizer.render_box_overlay(analysis_image, ranked_scores, topk=topk)

    class_name = str(original_class)
    if categories and 0 <= original_class < len(categories):
        class_name = categories[original_class]

    all_regions = build_region_payload(ranked_scores)
    top_regions = all_regions[:topk]

    return {
        "device": device,
        "class_index": original_class,
        "class_name": class_name,
        "confidence": original_conf,
        "analysis_image": pil_image_to_base64(analysis_image),
        "heatmap_overlay_image": pil_image_to_base64(heatmap_overlay),
        "box_overlay_image": pil_image_to_base64(box_overlay),
        "top_regions": top_regions,
        "all_regions": all_regions,
    }

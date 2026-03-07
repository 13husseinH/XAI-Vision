import argparse

import torch
from PIL import Image

from models.zoo import list_models, load_model, get_preprocess
from modules.masker import ImageMasker
from modules.scorer import ImageScorer


def predict_top1(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        prob = torch.softmax(output, dim=1)
        conf, cls = torch.max(prob, dim=1)
    return int(cls.item()), float(conf.item())


def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    model, weights = load_model(args.model, pretrained=(not args.no_pretrained), device=device)
    preprocess = get_preprocess(weights)

    pil_image = Image.open(args.image).convert("RGB")
    image = preprocess(pil_image).to(device)

    original_class, original_conf = predict_top1(model, image)

    masker = ImageMasker(
        fill_mode=args.fill_mode,
        constant_value=args.constant_value,
        noise_std=args.noise_std,
    )
    masked_results = masker.generate_all(image, grid_size=args.grid_size)

    scorer = ImageScorer()
    scores = scorer.score(model, image, masked_results)

    ranked = sorted(scores, key=lambda x: x["importance"], reverse=True)
    top_results = ranked[: args.topk]

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Original top-1 class index: {original_class}")
    print(f"Original top-1 confidence: {original_conf:.4f}")
    print(f"Masked samples generated: {len(masked_results)}")
    print(f"Top {len(top_results)} important regions:")

    for i, item in enumerate(top_results, start=1):
        y1, y2, x1, x2 = item["bbox"]
        importance = item["importance"]
        print(
            f"{i}. bbox=(y:{y1}-{y2}, x:{x1}-{x2}) | importance={importance:.4f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Run masking-based XAI scoring on one image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=list_models(),
        help="Backbone model name",
    )
    parser.add_argument("--grid-size", type=int, default=3, help="Grid size for grid masking")
    parser.add_argument("--topk", type=int, default=5, help="How many top regions to print")

    parser.add_argument(
        "--fill-mode",
        default="constant",
        choices=["constant", "mean", "noise"],
        help="Mask fill strategy",
    )
    parser.add_argument(
        "--constant-value",
        type=float,
        default=0.0,
        help="Fill value when fill_mode=constant",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.15,
        help="Noise standard deviation when fill_mode=noise",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly initialized model weights",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)

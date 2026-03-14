from torchvision import models


_SUPPORTED_MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.DEFAULT,
    ),
}


def list_models():
    return sorted(_SUPPORTED_MODELS.keys())


def load_model(name="resnet18", pretrained=True, device="cpu"):
    if name not in _SUPPORTED_MODELS:
        available = ", ".join(list_models())
        raise ValueError(f"Unsupported model '{name}'. Available: {available}")

    constructor, default_weights = _SUPPORTED_MODELS[name]

    if pretrained:
        try:
            model = constructor(weights=default_weights)
            weights_used = default_weights
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained weights for '{name}'."
            ) from exc
    else:
        model = constructor(weights=None)
        weights_used = None

    model = model.to(device)
    model.eval()
    return model, weights_used


def get_preprocess(weights):
    if weights is not None:
        return weights.transforms()

    # Fallback transform compatible with common ImageNet backbones.
    return models.ResNet18_Weights.DEFAULT.transforms()


def get_categories(weights):
    if weights is None:
        return None

    return weights.meta.get("categories")

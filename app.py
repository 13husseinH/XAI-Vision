import streamlit as st
import torch
from PIL import Image

from models.zoo import get_categories, get_preprocess, list_models, load_model
from modules.masker import ImageMasker
from modules.scorer import ImageScorer
from modules.visualizer import ImportanceVisualizer


st.set_page_config(page_title="XAI-Vision", layout="wide")


@st.cache_resource
def load_model_bundle(model_name, device):
    model, weights = load_model(model_name, pretrained=True, device=device)
    preprocess = get_preprocess(weights)
    categories = get_categories(weights)
    return model, preprocess, categories


def predict_top1(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        prob = torch.softmax(output, dim=1)
        conf, cls = torch.max(prob, dim=1)
    return int(cls.item()), float(conf.item())


def analyze_image(image, model_name, grid_size, fill_mode, constant_value, noise_std, topk):
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

    return {
        "device": device,
        "weights_loaded": True,
        "class_index": original_class,
        "class_name": class_name,
        "confidence": original_conf,
        "scores": ranked_scores,
        "overlay": overlay,
        "analysis_image": analysis_image,
    }


def main():
    st.title("XAI-Vision")
    st.write(
        "Upload an image, choose a pretrained model, and inspect which regions most affect the model's confidence."
    )

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Model", list_models(), index=0)
        grid_size = st.slider("Grid Size", min_value=2, max_value=6, value=3, step=1)
        topk = st.slider("Top Regions", min_value=1, max_value=10, value=5, step=1)
        fill_mode = st.selectbox("Fill Mode", ["constant", "mean", "noise"], index=0)
        constant_value = st.slider("Constant Fill Value", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        noise_std = st.slider("Noise Std", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
        run_analysis = st.button("Run Analysis", type="primary")

    uploaded_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.info("Upload a PNG or JPG image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if not run_analysis:
        st.caption("Adjust settings in the sidebar, then click Run Analysis.")
        return

    try:
        with st.spinner("Running masking analysis..."):
            results = analyze_image(
                image=image,
                model_name=model_name,
                grid_size=grid_size,
                fill_mode=fill_mode,
                constant_value=constant_value,
                noise_std=noise_std,
                topk=topk,
            )
    except RuntimeError as exc:
        st.error(str(exc))
        return

    st.subheader("Prediction")
    st.write(
        f"Top prediction: {results['class_name']} (class {results['class_index']}) with confidence {results['confidence']:.4f}"
    )
    st.caption(f"Inference device: {results['device']} | pretrained weights loaded: {results['weights_loaded']}")

    left, right = st.columns(2)
    with left:
        st.image(results["analysis_image"], caption="Analyzed View (224x224)", use_container_width=True)
    with right:
        st.image(results["overlay"], caption="Importance Overlay", use_container_width=True)

    st.subheader("Top Regions")
    for rank, item in enumerate(results["scores"][:topk], start=1):
        y1, y2, x1, x2 = item["bbox"]
        st.write(
            f"{rank}. bbox=(y:{y1}-{y2}, x:{x1}-{x2}) | importance={item['importance']:.4f}"
        )


if __name__ == "__main__":
    main()

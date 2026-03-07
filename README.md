# XAI-Vision
Deep Learning models are often "Black Boxes"; we see what they predict, but we don't know *why*. **XAI-Vision** is a diagnostic suite designed to "show" the internal logic of AI.

 [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
 
XAI-Vision works by **systematically masking regions of an image** and measuring how the model’s prediction confidence changes.  
If hiding a region causes a large confidence drop, that region is likely **important for the model's decision**.

---

## How It Works

1. Load a pretrained vision model (e.g., `resnet18`, `mobilenet_v3_small`).
2. Preprocess the input image.
3. Generate masked variants of the image (grid, borders, corners, center).
4. Run inference on each masked image.
5. Measure the confidence drop compared to the original prediction.
6. Rank regions by importance.

<img width="1394" height="768" alt="best of best" src="https://github.com/user-attachments/assets/7260d19d-6fed-47db-b229-61311db14d7e" />


In this example:

- **Model:** ResNet18 (ImageNet pretrained)
- **Original confidence:** 0.8285
- **Most important region:** `(y:0–74, x:74–148)`
- **Importance score:** `0.5193`

## Mask Types

<img width="1536" height="1024" alt="ex imagee" src="https://github.com/user-attachments/assets/58945392-adc9-4b9b-b9ce-5cac5ad1ed93" />

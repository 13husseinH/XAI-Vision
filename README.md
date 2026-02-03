# XAI-Vision
Deep Learning models are often "Black Boxes"; we see what they predict, but we don't know *why*. **XAI-Vision** is a diagnostic suite designed to "show" the internal logic of AI.

 [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
 
## XAI-Vision Suite is:
 An Image Processor (to mask/corrupt images).
 An AI Interrogator (to check different models like ResNet or ViT).
 A Visualizer (to create the heatmap UI).
 A Scorer (to calculate if the model is cheating or is bias).

## 📂 Project Structure
```text
XAI-Vision/
├── app.py                # Main UI & Dashboard
├── modules/              # Logic Modules
│   ├── masker.py         # Image Ablation
│   ├── scorer.py         # Bias Quantification
│   └── visualizer.py     # Heatmap Generation
├── models/               # AI Engine
│   └── zoo.py            # Model Loading (ResNet, ViT, etc.)
└── requirements.txt      # Dependencies

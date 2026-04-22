# 🦋 Butterfly Species Classification & Identifier

An interactive Streamlit dashboard and robust deep learning pipeline for fine-grained butterfly species classification across 100 species.

**The Problem:** Snap a photo on a forest trek → get the species name, conservation status, and a fun fact.
**The Solution:** A lightweight, highly efficient transfer-learning model deployed via Streamlit, providing Top-3 predictions with rich biological context (IUCN status, Wikipedia summaries, and fun facts).

---

## ✨ Features

- **Interactive Streamlit Dashboard:** Upload an image (JPG/PNG/WEBP) and instantly receive the Top-3 species predictions with confidence scores.
- **Rich Species Metadata:** Displays real-time IUCN Conservation Status badges, Wikipedia summaries, and curated fun facts for the predicted species.
- **Zero-Shot Inference (CLIP):** Built and evaluated a prompt-ensembled zero-shot pipeline using `openai/clip-vit-base-patch32`.
- **High-Efficiency Fine-Tuning (EfficientNet):** Achieved **96.60% Test Top-1 Accuracy** using a strategic two-phase fine-tuning approach on `efficientnet_b0`, requiring only 31% of the model's parameters to be updated.
- **Model Explainability:** Includes Grad-CAM heatmaps to visualize where the model "looks" (validating it looks at the wings/thorax, not the background) and UMAP projections to visualize the learned feature manifold.

---

## 🧠 Architectures & Methodology

This project explores two distinct paradigms for image classification:

### 1. Zero-Shot Learning (CLIP)
- **Model:** `openai/clip-vit-base-patch32`
- **Approach:** Used pre-trained vision-language contrastive representations. We ensemble multiple text prompts (e.g., `"A photo of a {species}, a type of butterfly."`, `"A close-up field photo of a {species} butterfly."`) to create robust text anchors for each of the 100 classes.
- **Pros:** No training required, highly generalized.

### 2. Supervised Fine-Tuning (EfficientNet-B0)
- **Model:** `efficientnet_b0` (via PyTorch Image Models - `timm`)
- **Backbone Initialization:** ImageNet pre-trained weights.
- **Data Augmentation:** Random resized crops, heavy horizontal flipping (leveraging biological bilateral symmetry), mild vertical flipping, color jittering (for varying field lighting), and rotation.

#### 🔬 The Two-Phase Fine-Tuning Strategy
Instead of blindly fine-tuning the entire model (which is prone to overfitting and catastrophic forgetting of low-level features), we used a structured two-phase approach:

1. **Phase 1: Linear Probe (Warm-up)**
   - **Action:** Freeze the entire convolutional backbone. Train *only* the new linear classification head (100 classes).
   - **Duration:** 5 epochs.
   - **Trainable Params:** ~0.13M parameters.
   - **Result:** Reached ~89.8% validation accuracy. This proved that generic ImageNet features are deeply transferable to butterfly morphology. It also warms up the random classifier weights.

2. **Phase 2: Fine-Tune Last Block**
   - **Action:** Unfreeze the deep, semantic features of the last MBConv block (`blocks.6`) along with the warmed-up head. 
   - **Duration:** 10 epochs with a 10x smaller learning rate.
   - **Trainable Params:** ~1.27M parameters (just 31% of the total model).
   - **Result:** Reached **96.60% Test Top-1 Accuracy** and **99.40% Test Top-3 Accuracy**.

#### 📊 Ablation Study
We conducted an ablation study to validate this choice against a "Full Fine-Tune" (unfreezing all 4.14M parameters from scratch):

| Method | Trainable Params | Epochs | Test Top-1 | Test Top-3 | Macro F1 |
|---|---:|:---:|:---:|:---:|:---:|
| **Baseline (Author H5)** | 0 | 0 | 97.60% | 99.60% | — |
| **A — Linear Probe** | 0.13M | 5 | ~89% | — | — |
| **B — Fine-Tune Last Block ✅** | 1.27M | 5+10 | **96.60%** | **99.40%** | **96.45%** |
| **C — Full Fine-Tune** | 4.14M | 10 | 96.60% | 99.20% | 96.53% |

**Conclusion:** Unfreezing just the last block (Method B) matches the Top-1 performance of full fine-tuning (Method C) while saving massive compute, reducing parameters by 3.2x, and minimizing validation overfitting. It also slightly improves Top-3 accuracy.

---

## 🚀 Running the App Locally

### Prerequisites
Make sure you have Python 3.9+ installed.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Assignment-3
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Upload an image!**
   The app will automatically route to your local browser at `http://localhost:8501`. Drop a field photo of a butterfly to see the predictions.

---

## 📂 Repository Structure

```text
├── app.py                             # Main Streamlit dashboard script
├── requirements.txt                   # Dependency list
├── Zero-Shot-CLIP.ipynb               # Jupyter notebook for CLIP zero-shot pipeline
├── FineTune-EfficientNet.ipynb        # Jupyter notebook for the EfficientNet pipeline
└── Kaggle_run/
    ├── efficientnet_butterfly.pth     # Saved PyTorch model weights (Phase 2)
    ├── class_names.json               # 100-species list
    └── SMAI_Assignment_3/
        ├── cached_species_info.json   # Scraped Wikipedia & IUCN metadata
        └── species_images/            # Reference photos used by the app
```

### Note on Artifacts
Due to Github size limitations, the `.pth` weights file and cached images might need to be downloaded via Kaggle or Git LFS depending on your setup. The `Kaggle_run` folder contains the output artifacts of the Kaggle notebooks.

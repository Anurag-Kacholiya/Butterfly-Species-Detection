This is a wonderfully structured mini-project. The constraints—especially relying on zero-shot inference or light fine-tuning—push you to think deeply about data representations rather than just throwing compute at the problem. 

Given your focus on gaining deeper insights into the data and the species, you can do some really clever things here, especially by bridging your understanding of contextual embeddings and vision. Here are a few innovative methods and architectural strategies that will make your app stand out and give you excellent material for the "Ablation" and "Results" sections of your technical report.

### 1. The CLIP Pathway: Prompt Ensembling & Space Exploration
CLIP uses Byte Pair Encoding (BPE) to tokenize text and a Vision Transformer (ViT) to process images, projecting both into a shared embedding space. You can exploit this shared space for unique insights.

* **Prompt Ensembling (Great for Ablation):** Instead of calculating the similarity between the image and a single text prompt like `"A photo of a {species}"`, create an ensemble of prompts. 
    * `"A photo of a {species}, a type of butterfly."`
    * `"A close-up macro shot of a {species} butterfly."`
    * `"A {species} butterfly resting on a flower."`
    Extract the text embeddings for all these variations, average them, and normalize the result. This creates a much more robust text anchor in the vector space. Comparing Single Prompt vs. Ensembled Prompts is a perfect ablation study for your report.
* **Zero-Shot Error Analysis via Nearest Neighbors:** If CLIP misclassifies a butterfly, look at the top-3 predictions. Are they visually similar? You can calculate the cosine similarity between the embeddings of the misclassified image and the text embeddings of all 100 species to map out *why* the model was confused.

### 2. The EfficientNet Pathway: Visual Interpretability
If you choose the `efficientnet_b0` fine-tuning route, your primary innovation should be explainability. 

* **Grad-CAM (Gradient-weighted Class Activation Mapping):** This is highly recommended for your Streamlit app. When a user uploads a photo, don't just output the prediction; generate a heatmap overlay showing *where* the model is looking. For butterflies, you want to see if the model is correctly activating on the wing patterns, or if it's "cheating" by looking at the background (e.g., associating a specific species with a specific type of leaf it was photographed on). This provides massive insight into dataset biases.
* **Linear Probing:** Freeze the entire EfficientNet model and only train a new classification head (a single linear layer) on top of the extracted features. This is mathematically lightweight and trains in minutes.

### 3. Data Insights: Embedding Clustering
You can generate incredible insights for your technical report without training a single weight.

* **t-SNE or UMAP Visualizations:** Pass a subset of your 100-species Kaggle dataset through the frozen CLIP vision encoder (or frozen EfficientNet) to extract the image embeddings. Use UMAP or t-SNE to reduce these high-dimensional vectors to 2D and plot them. 
* **What this tells you:** Do the species cluster neatly? You might find that butterflies from the same biological family (like *Papilionidae* or swallowtails) naturally cluster together purely based on visual morphology, even if the model has no taxonomic data.

### 4. Smart Caching and LLM Integration
The requirement to cache the IUCN status and fun facts using Gemini is a great touch. 

* **Pre-computation:** Don't do this at runtime. Write a short Python script that iterates through the 100 class names from the Kaggle dataset. Prompt Gemini to output a strict JSON structure containing the `species_name`, `iucn_status` (e.g., "Least Concern", "Endangered"), and `fun_fact`. Save this as `species_metadata.json` in your repo.
* **Streamlit UI Integration:** When the model outputs its top-3 predictions, your Streamlit app just does a fast dictionary lookup in the JSON file to pull the badge data and the Wikipedia summary link.

### Deliverables Strategy
* **The Ablation Section:** This is where you score high marks. If you use CLIP, ablate the prompt structures. If you use EfficientNet, ablate the learning rate, or compare a frozen backbone (linear probe) vs. unfreezing the last convolutional block.
* **The App UI:** Keep the Streamlit layout clean. Use `st.columns` to display the uploaded image on the left and the top-3 predictions with progress bars (representing confidence probabilities) on the right.

We have to implement both the zero-shot elegance of CLIP, and the feature-extraction and fine-tuning control of EfficientNet methods in two separate notebooks named Zero-Shot-CLIP.ipynb and FineTune-EfficientNet.ipynb. And then we have to create a streamlit app that will use both the models and compare their performance.
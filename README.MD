# Real-Time ASL Alphabet Recognition (Nebula)

**Demo recording**: [Watch the demo video](https://drive.google.com/file/d/1ouYzJDXAJR91enOn2zaWjRjcyUDQggCY/view?usp=sharing)

A lightweight, browser-based demo that lets you sign ASL letters in front of your webcam and see them live in your browser via Streamlit.

---

## 📋 Overview

This repo contains:

-   **Streamlit demo** (`streamlit_app.py`) for real-time ASL → Text → Speech

-   **Webcam test script** (`webcam.py`) to verify your model locally

-   **Training notebooks**:

    -   `ASL_colab.ipynb` (Kaggle/Colab training version)

    -   `main.ipynb` (offline training pipeline)

-   **Pretrained models**:

    -   `asl_best_model.h5` (best offline model)

    -   `asl_colab.h5` (Colab-trained model)

    -   `asl.h5` (final offline export)

-   **Dataset reference**: [Kaggle ASL-Alphabet (29 classes)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)


---

## 🚀 Quick Start

1.  **Clone & enter**

    ```bash
    git clone https://github.com/AnshBhardwaj1/nebula_project.git
    cd nebula_project-main
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the ASL dataset** (for retraining only)

    ```bash
    # From Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    unzip into ./dataset/asl_alphabet_train/
    ```

4.  **Pick your model**

    ```bash
    cp asl_best_model.h5 streamlit_app.py
    ```


---

## 🎬 Run the Streamlit Demo

```bash
streamlit run streamlit_app.py
```

Open your browser at the URL it prints (usually `http://localhost:8501`).
You’ll see your live webcam, a 5-frame majority-voting buffer, and TTS audio output.

**Live demo recording**: [Watch the demo video](https://drive.google.com/file/d/1ouYzJDXAJR91enOn2zaWjRjcyUDQggCY/view?usp=sharing)

<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🛠 How It Works

1.  **Capture & Crop**
    OpenCV captures frames and crops a fixed ROI (configurable in code).

2.  **Preprocess**
    Resize to 64×64 and normalize pixel values.

3.  **Inference**
    Run the Keras model on each ROI. Use a deque of length 5 for majority voting to stabilize jittery predictions.

4.  **Display & Speak**
    Streamlit shows live frames with overlay text. `pyttsx3` speaks out the buffered text.


You can tweak the buffer size, ROI coordinates, or swap in any of your three `.h5` models by editing the top of `streamlit_app.py`.

---
**## 📂 Repo Structure**

```
nebula_project-main/
├── ASL_colab.ipynb       # Colab training notebook
├── main.ipynb            # Offline training notebook
├── asl_best_model.h5     # Best offline model
├── asl_colab.h5          # Colab-trained model
├── asl.h5                # Final offline export
├── dataset/              # (for retraining) unzipped Kaggle data
├── streamlit_app.py      # Streamlit realtime demo
├── webcam.py             # Quick local test script
├── guidelines.md         # Nebula submission guidelines
├── requirements.txt      # pip dependencies
└── README.md             # ← this file
```

---

**## 🔄 Retrain or Improve**

If you want to retrain or swap in a new model:

1. Modify & run `ASL_colab.ipynb` (on Colab) or `main.ipynb` (offline).
2. Save your new `.h5` into the repo root.
3. Update line 15 of `streamlit_app.py` to point at your new model file.
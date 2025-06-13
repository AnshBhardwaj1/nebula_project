# Real-Time ASL Alphabet Recognition

A lightweight, browser-based demo that captures your webcam feed, recognizes American Sign Language (ASL) fingerspelling letter by letter, displays the accumulated text buffer (up to 5 characters), and speaks it out loud.

---

## 📋 Overview

This project uses an OpenCV + TensorFlow/Keras pipeline to:

1. Capture live video from your webcam  
2. Crop a region of interest (ROI) and preprocess frames  
3. Run a trained ASL-alphabet classifier model on each ROI  
4. Buffer the last 5 predicted characters for stability  
5. Render the live video + overlay text in your browser  
6. Convert the buffered text to speech (TTS)  

We’ve provided both a **Flask** and a **Streamlit** demo—pick whichever you prefer to spin up quickly.

---

## 🚀 Quick Start

### 1. Clone & enter repo  
```bash
git clone https://github.com/AnshBhardwaj1/nebula_project.git
cd asl-demo

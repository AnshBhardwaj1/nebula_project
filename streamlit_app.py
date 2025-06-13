import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.sidebar.title("Settings")
MODEL_PATH = st.sidebar.text_input("Model path", "asl.h5")
BUFFER_SIZE = st.sidebar.slider("Buffer size", 5, 30, 15)

@st.cache_resource
def load_asl_model(path):
    return load_model(path)
model = load_asl_model(MODEL_PATH)
labels = [chr(65 + i) for i in range(26)] + ["", "", " "]

from collections import deque, Counter
pred_buffer = deque(maxlen=BUFFER_SIZE)

st.title("🖐️ ASL → Text → Speech")
FRAME_PLACEHOLDER = st.empty()
TEXT_PLACEHOLDER  = st.empty()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("Cannot open webcam")
    st.stop()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    img = cv2.resize(roi, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    letter = labels[np.argmax(pred)]
    pred_buffer.append(letter)
    current_char = Counter(pred_buffer).most_common(1)[0][0]

    FRAME_PLACEHOLDER.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    TEXT_PLACEHOLDER.markdown(f"**Detected:** {current_char}")

    if st.session_state.get("stop", False):
        break

cap.release()

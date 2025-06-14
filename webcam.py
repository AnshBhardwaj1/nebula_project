
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from collections import deque, Counter
import win32com.client as winc1

def main():
    
    speaker = winc1.Dispatch("SAPI.SpVoice")

    model = load_model('asl_best_model.h5')

    labels = [chr(65 + i) for i in range(26)] + ['', '', ' ']

    buffer_size = 15
    pred_buffer = deque(maxlen=buffer_size)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    accumulated_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img = cv2.resize(roi, (64, 64))
        img = img.astype('float32') / 255.0
        x_input = np.expand_dims(img, axis=0)

        probs = model.predict(x_input, verbose=0)[0]
        pred_class = np.argmax(probs)
        pred_buffer.append(pred_class)

        if len(pred_buffer) == buffer_size:
            most_common = Counter(pred_buffer).most_common(1)[0][0]
            char = labels[most_common]
            accumulated_text += char
            pred_buffer.clear()

        last_five = accumulated_text[-5:]
        cv2.putText(frame, f"Text: {last_five}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    final_text = accumulated_text[-5:]
    print("Final:", final_text)
    speaker.Speak(final_text)

if __name__ == "__main__":
    main()

# This is the part of the code responsible for Video Capturing and grayscaling and resizing.

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def load_labels(labels_path):
    # Single source of truth for class names; fail loudly rather than fall back.
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    with labels_file.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Labels file must contain a non-empty list: {labels_file}")
    return labels


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized / 255.0
    return np.reshape(normalized, (1, 128, 128, 1))


def main():
    parser = argparse.ArgumentParser(description="Run the ISL webcam predictor")
    parser.add_argument("--model-path", default="ISL_Interpreter.keras", help="Path to the trained model")
    parser.add_argument("--labels", default="labels.json", help="JSON file with the class labels")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence required for a prediction")
    args = parser.parse_args()

    model = load_model(args.model_path)
    labels = load_labels(args.labels)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera frame. Exiting.")
            break

        reshaped = preprocess_frame(frame)
        prediction = model.predict(reshaped, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < args.threshold:
            sign = "uncertain"
        else:
            sign = labels[predicted_class]

        cv2.putText(frame, f"{sign} ({confidence:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

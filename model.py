import argparse
import json
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

DEFAULT_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P"
]
IMG_SIZE = 128


def load_labels(labels_path):
    labels_file = Path(labels_path)
    if labels_file.exists():
        with labels_file.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        if isinstance(labels, list) and labels:
            return labels
    return DEFAULT_LABELS


def create_training_data(data_dir, categories):
    training_data = []
    skipped = []

    for category in categories:
        class_dir = Path(data_dir) / category
        if not class_dir.exists():
            warnings.warn(f"Missing directory for class '{category}': {class_dir}")
            continue

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped.append(str(image_path))
                continue

            resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized, categories.index(category)])

    if not training_data:
        raise FileNotFoundError(f"No usable images found under {data_dir}")

    random.shuffle(training_data)

    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(y, dtype=np.int32), skipped


def build_model(num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the ISL CNN")
    parser.add_argument("--data-dir", default="ISL_Dataset", help="Folder containing one image directory per class")
    parser.add_argument("--labels", default="labels.json", help="JSON file with the class labels")
    parser.add_argument("--output", default="ISL_Interpreter.keras", help="Location to save the trained model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    categories = load_labels(args.labels)
    X, y, skipped = create_training_data(args.data_dir, categories)
    if skipped:
        print(f"Skipped unreadable images: {len(skipped)}")

    X = X / 255.0
    model = build_model(len(categories))

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.1, callbacks=[lr_scheduler])

    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

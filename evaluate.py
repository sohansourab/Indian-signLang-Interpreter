import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from model import build_model, create_training_data, load_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ISL CNN")
    parser.add_argument("--data-dir", default="ISL_Dataset", help="Folder containing one image directory per class")
    parser.add_argument("--labels", default="labels.json", help="JSON file with class labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to reserve for testing")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    X, y, _ = create_training_data(args.data_dir, labels)
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    model = build_model(len(labels))
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(json.dumps({"test_loss": float(loss), "test_accuracy": float(accuracy)}, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model

from model import IMG_SIZE, load_labels


def load_split(split_dir, labels):
    split_dir = Path(split_dir)
    images = []
    targets = []
    for index, label in enumerate(labels):
        class_dir = split_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
            targets.append(index)
    X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(targets, dtype=np.int32)
    return X, y


def evaluate_split(model, split_dir, labels):
    X, y = load_split(split_dir, labels)
    predictions = model.predict(X, batch_size=64, verbose=0).argmax(axis=1)
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(y, predictions, labels=indices, zero_division=0)
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    result = {
        "num_images": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
    }
    return result, y, predictions


def save_confusion_matrix(y, predictions, labels, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y, predictions, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix (Testing)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved ISL CNN")
    parser.add_argument("--model-path", default="ISL_Interpreter.keras", help="Saved model to evaluate")
    parser.add_argument("--labels", default="labels.json", help="JSON file with class labels")
    parser.add_argument("--test-dir", default="Testing", help="Held-out test folder")
    parser.add_argument("--val-dir", default="Validation", help="Validation folder")
    parser.add_argument("--output-dir", default="results", help="Folder for metrics and plots")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded with the metrics")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model = load_model(args.model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_path": str(args.model_path),
        "labels": labels,
        "seed": args.seed,
        "note": "Testing numbers include known duplicate images; see results/duplicate_report.json and splits/clean_test.txt.",
        "splits": {},
    }

    for name, split_dir in (("Testing", args.test_dir), ("Validation", args.val_dir)):
        result, y, predictions = evaluate_split(model, split_dir, labels)
        metrics["splits"][name] = result
        print(f"{name}: n={result['num_images']} accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
        if name == "Testing":
            save_confusion_matrix(y, predictions, labels, output_dir / "confusion_matrix.png")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

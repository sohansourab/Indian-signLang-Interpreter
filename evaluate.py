import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model

from model import IMG_SIZE, load_labels


def load_manifest(manifest_path):
    # Manifest of image paths (relative to the repo root) to keep out of evaluation.
    return {line.strip() for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()}


def load_split(split_dir, labels, include=None):
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
            if include is not None and image_path.as_posix() not in include:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
            targets.append(index)
    X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(targets, dtype=np.int32)
    return X, y


def evaluate_split(model, split_dir, labels, include=None):
    X, y = load_split(split_dir, labels, include)
    predictions = model.predict(X, batch_size=64, verbose=0).argmax(axis=1)
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(y, predictions, labels=indices, zero_division=0)
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "accuracy": float(recall[i]),
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


def save_confusion_matrix(y, predictions, labels, output_path, title="Confusion matrix (Testing)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y, predictions, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
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
    parser.add_argument("--exclude-duplicates", default=None, help="Manifest of Testing images to exclude for the clean-test evaluation")
    parser.add_argument("--output", default="results/metrics.json", help="Where to write the metrics")
    parser.add_argument("--confusion-matrix", default="results/confusion_matrix.png", help="Where to write the confusion matrix")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded with the metrics")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model = load_model(args.model_path)
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_path": str(args.model_path),
        "labels": labels,
        "seed": args.seed,
        "note": "Testing numbers include known duplicate images; see results/duplicate_report.json and splits/clean_test.txt.",
        "splits": {},
    }

    if args.exclude_duplicates:
        manifest = load_manifest(args.exclude_duplicates)
        metrics["exclude_duplicates_manifest"] = str(args.exclude_duplicates)
        plan = [("Testing", args.test_dir, None), ("Testing_clean", args.test_dir, manifest), ("Validation", args.val_dir, None)]
    else:
        plan = [("Testing", args.test_dir, None), ("Validation", args.val_dir, None)]

    for name, split_dir, include in plan:
        result, y, predictions = evaluate_split(model, split_dir, labels, include)
        metrics["splits"][name] = result
        print(f"{name}: n={result['num_images']} accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
        if name == "Testing":
            save_confusion_matrix(y, predictions, labels, args.confusion_matrix)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

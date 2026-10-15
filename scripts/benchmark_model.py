import argparse
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def find_sample(sample_dir):
    sample_dir = Path(sample_dir)
    for class_dir in sorted(d for d in sample_dir.iterdir() if d.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                return path
    raise FileNotFoundError(f"No sample images under {sample_dir}")


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.reshape(normalized, (1, 128, 128, 1))


def main():
    parser = argparse.ArgumentParser(description="Benchmark in-process single-frame inference")
    parser.add_argument("--model-path", default="ISL_Interpreter.keras", help="Saved model to benchmark")
    parser.add_argument("--sample-dir", default="Testing", help="Folder to draw a sample image from")
    parser.add_argument("--requests", type=int, default=100, help="Number of measured frames")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up frames to discard")
    parser.add_argument("--output", default="results/model_latency.txt", help="Where to write the report")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow.keras.models import load_model

    model = load_model(args.model_path)
    frame = cv2.imread(str(find_sample(args.sample_dir)))
    if frame is None:
        raise FileNotFoundError("Could not read the sample image")
    batch = preprocess(frame)

    for _ in range(args.warmup):
        model.predict(batch, verbose=0)

    model_only = []
    for _ in range(args.requests):
        start = time.perf_counter()
        model.predict(batch, verbose=0)
        model_only.append((time.perf_counter() - start) * 1000.0)

    end_to_end = []
    for _ in range(args.requests):
        start = time.perf_counter()
        model.predict(preprocess(frame), verbose=0)
        end_to_end.append((time.perf_counter() - start) * 1000.0)

    model_only = np.array(model_only)
    end_to_end = np.array(end_to_end)
    lines = [
        "In-process single-frame inference (CPU)",
        f"model: {args.model_path}",
        f"measured_frames: {args.requests}",
        f"warmup_frames: {args.warmup}",
        f"model_only_median_ms: {float(np.median(model_only)):.2f}",
        f"model_only_p95_ms: {float(np.percentile(model_only, 95)):.2f}",
        f"model_only_fps: {1000.0 / float(model_only.mean()):.2f}",
        f"end_to_end_median_ms: {float(np.median(end_to_end)):.2f}",
        f"end_to_end_p95_ms: {float(np.percentile(end_to_end, 95)):.2f}",
        f"end_to_end_fps: {1000.0 / float(end_to_end.mean()):.2f}",
        f"cpu: {platform.processor() or platform.uname().machine}",
        f"cpu_count_logical: {os.cpu_count()}",
        f"os: {platform.platform()}",
        f"python: {platform.python_version()}",
    ]
    report = "\n".join(lines) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

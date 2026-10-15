import argparse
import base64
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def wait_for_server(base_url, path="/", attempts=120, delay=1.0):
    url = base_url.rstrip("/") + path
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(delay)
    return False


def collect_samples(images_dir, count):
    images_dir = Path(images_dir)
    class_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
    per_class = [sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) for d in class_dirs]
    per_class = [items for items in per_class if items]
    samples = []
    index = 0
    while len(samples) < count and per_class:
        for items in per_class:
            if len(samples) >= count:
                break
            samples.append(items[index % len(items)])
        index += 1
    return samples


def encode(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def post_predict(url, image_data):
    payload = json.dumps({"image": image_data}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
    return (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark the FastAPI /predict endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running app")
    parser.add_argument("--images-dir", default="Testing", help="Folder of sample images (one subfolder per class)")
    parser.add_argument("--requests", type=int, default=100, help="Number of measured requests")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up requests to discard")
    parser.add_argument("--output", default="results/api_latency.txt", help="Where to write the report")
    args = parser.parse_args()

    if not wait_for_server(args.base_url):
        raise SystemExit(f"Server at {args.base_url} did not become ready")

    samples = collect_samples(args.images_dir, args.requests + args.warmup)
    if len(samples) < args.requests + args.warmup:
        raise SystemExit(f"Not enough sample images: found {len(samples)}")

    url = args.base_url.rstrip("/") + "/predict"
    latencies = []
    for i, path in enumerate(samples):
        elapsed = post_predict(url, encode(path))
        if i >= args.warmup:
            latencies.append(elapsed)

    latencies = np.array(latencies)
    requests_per_second = 1000.0 / float(latencies.mean())

    lines = [
        "FastAPI /predict endpoint benchmark",
        f"base_url: {args.base_url}",
        f"measured_requests: {int(latencies.size)}",
        f"warmup_requests: {args.warmup}",
        f"median_latency_ms: {float(np.median(latencies)):.2f}",
        f"p95_latency_ms: {float(np.percentile(latencies, 95)):.2f}",
        f"max_latency_ms: {float(latencies.max()):.2f}",
        f"requests_per_second: {requests_per_second:.2f}",
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

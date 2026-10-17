import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

IMG_SIZE = 128


def main():
    parser = argparse.ArgumentParser(description="Benchmark the saved model on a GPU if one is available")
    parser.add_argument("--model-path", default="ISL_Interpreter.keras", help="Saved model to benchmark")
    parser.add_argument("--output", default=None, help="Optional file to write the result")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        message = "GPU inference: [NOT MEASURED] - no GPU visible to this TensorFlow build."
        print(message)
        if args.output:
            Path(args.output).write_text(message + "\n", encoding="utf-8")
        return

    from tensorflow.keras.models import load_model

    model = load_model(args.model_path)
    x = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    # Warm up
    for _ in range(3):
        model.predict(x, verbose=0)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(x, verbose=0)
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(0.95 * (len(times) - 1))]
    result = (
        f"GPU inference (device: {gpus[0].name}): "
        f"median={median:.2f} ms  p95={p95:.2f} ms  "
        f"throughput={1000.0 / median:.2f} FPS"
    )
    print(result)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
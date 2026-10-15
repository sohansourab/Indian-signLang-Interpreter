import argparse
import ctypes
import os
import platform
import subprocess
from pathlib import Path


def cpu_model():
    try:
        output = subprocess.check_output(
            ["wmic", "cpu", "get", "name"], text=True, stderr=subprocess.DEVNULL
        )
        names = [line.strip() for line in output.splitlines() if line.strip() and line.strip() != "Name"]
        if names:
            return names[0]
    except Exception:
        pass
    return platform.processor() or platform.uname().machine


def total_ram_gb():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullTotalPhys / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description="Record the runtime environment and model details")
    parser.add_argument("--model-path", default="ISL_Interpreter.keras", help="Saved model file")
    parser.add_argument("--output", default="results/environment.txt", help="Where to write the report")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import cv2
    import fastapi
    import keras
    import tensorflow as tf

    lines = [
        "Environment and model details",
        f"os: {platform.platform()}",
        f"os_version: {platform.version()}",
        f"cpu: {cpu_model()}",
        f"cpu_count_logical: {os.cpu_count()}",
        f"ram_gb: {total_ram_gb():.2f}",
        f"python: {platform.python_version()}",
        f"tensorflow: {tf.__version__}",
        f"keras: {keras.__version__}",
        f"opencv: {cv2.__version__}",
        f"fastapi: {fastapi.__version__}",
    ]

    model_path = Path(args.model_path)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 ** 2)
        model = keras.models.load_model(str(model_path))
        lines.append(f"model_file: {model_path.name}")
        lines.append(f"model_size_mb: {size_mb:.2f}")
        lines.append(f"model_parameters: {model.count_params()}")
        lines.append(f"model_output_shape: {model.output_shape}")
    else:
        lines.append(f"model_file: {model_path.name} [NOT MEASURED - file missing]")

    report = "\n".join(lines) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

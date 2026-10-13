# Phase 0: Baseline freeze

## What was verified

- Repository root contains: `README.md`, `model.py`, `ISL_Interpreter.py`, `Images/`, `.github/`, and no `ISL_Dataset/` directory.
- Python runtime: `py -V` -> `Python 3.14.6`.
- TensorFlow import check: `py -c "import tensorflow as tf"` -> `ModuleNotFoundError: No module named 'tensorflow'`.
- OpenCV/NumPy import check was not possible in the current environment because TensorFlow import failed before the combined command reached those modules.

## Dataset status

- `ISL_Dataset` is not present in the current workspace.
- Because the dataset is absent, the exact class counts, image size, and per-class image totals remain unknown.
- Command to run on a machine with the dataset:

```bash
py -c "import os; d=r'.\\ISL_Dataset'; print({c: len(os.listdir(os.path.join(d,c))) for c in sorted(os.listdir(d))})"
```

## Version snapshot

```bash
py -V
py -c "import sys; print(sys.executable)"
py -c "import tensorflow as tf, cv2, numpy; print('TF', tf.__version__); print('CV2', cv2.__version__); print('NP', numpy.__version__)"
```

## Baseline run status

- Training run was not executed in this workspace because the environment is missing TensorFlow and the dataset folder is absent.
- Baseline accuracy is therefore: unknown.
- The original code still contains hardcoded Windows paths and a model-save call without an extension; those must be fixed before a meaningful model run can be expected.

## Summary

The project is not yet in a reproducible baseline state. The repo has the source files, but the runtime requirements and dataset needed for a real model run are not present here.

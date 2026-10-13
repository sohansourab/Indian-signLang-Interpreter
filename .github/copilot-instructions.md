# Copilot instructions: ISL Interpreter

Project: ISL fingerspelling recognizer. `model.py` trains a Keras CNN on grayscale 128x128 letter images and saves a model. `ISL_Interpreter.py` loads it and predicts live from the webcam. The owner is a student who must understand and explain every line.

## Environment
Python 3.14.6, TensorFlow/Keras not installed in the current environment, OpenCV not installed in the current environment, OS Windows.
Dataset layout: `<<data_dir>>/<LETTER>/*.jpg` (not committed to git).

## Current phase
Phase 0: baseline freeze. Only work on this phase.
Phases: 0 baseline, 1 clean and reproducible, 2 honest evaluation, 3 improve the model, 4 real-time inference, 5 FastAPI + web demo, 6 Docker/tests/deploy, 7 README and résumé packaging.

## Preservation rules (most important)
- Do NOT rewrite, reformat, rename or restructure existing files unless I explicitly ask.
- Prefer the smallest possible edit: change specific lines, not whole files. No drive-by cleanups.
- Keep the existing CNN architecture, variable names and file layout until I approve a change.
- Never delete code. If something looks unused or wrong, point it out with file and line numbers and ask me.
- One change per request. Before editing, list the exact lines you will touch and why, then wait for my "go".
- After editing, show a short diff summary and the command to test it.

## Honesty rules
- Don't assume dataset size, class list, accuracy or that a feature works. If unknown, say "unknown" and tell me how to check.
- Never invent or hardcode metrics. Every number in the README or docs must come from output I paste.

## Engineering rules
- No hardcoded absolute paths; use relative paths and argparse or a config file.
- No code that executes on import; use `if __name__ == "__main__":`.
- Class names come from a single `labels.json`, never copy-pasted lists.
- Save models in `.keras` format. Fixed random seeds (42) for splits and training.
- Change one thing at a time when improving the model and compare to the baseline on the same test split.
- Never commit datasets, trained models, logs, pickles, credentials or personal data.
- Don't swallow errors with bare `except: pass`; log skipped or unreadable images.
- Use respectful terminology: "deaf and hard-of-hearing", never "dumb".

## Code style
- Python 3, type hints on public functions, short docstrings, small functions.
- Comments explain why, because the owner must defend this code in an interview.
- Prefer standard libraries (tensorflow, opencv, numpy, scikit-learn, fastapi, pytest). Pin versions in requirements.txt.

## Known issues (already audited; fix only when asked)
- Hardcoded absolute Windows paths; training runs on import; pickle step; CATEGORIES duplicated.
- `model.save("ISL_Interpreter")` has no extension (fails on newer Keras).
- Webcam script squashes the whole frame to 128x128, has no confidence threshold, doesn't check the camera read.
- No test set or evaluation script.

## How to respond
- Before large edits, state which files you will change and why, then wait for confirmation.
- If something can't be verified without the dataset or the owner's machine, say so instead of guessing.

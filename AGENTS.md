# ISL Interpreter: agent rules

Project: ISL fingerspelling letter recognizer. Keras CNN (`model.py`), evaluation (`evaluate.py`), OpenCV webcam (`ISL_Interpreter.py`), FastAPI app (`web_app.py`), Docker. The owner is a student who must be able to explain every change in an interview.

Environment: Python 3.12.10, TensorFlow 2.16.2, OS Windows.
Dataset folders (never committed): `Training/`, `Validation/`, `Testing/`, each with one subfolder per letter.

## Autonomy
- Work end to end on the task I give you. Do not stop to ask for approval. Take the defaults stated in the task.
- Ask a question only if you are blocked by information you cannot find in the repo or by running a command.
- After finishing: run the verify commands, then `git add` the changed files and commit on the current branch with a clear message. Never push, never force, never change branches, never rewrite history.

## Hard limits (stop and report instead of proceeding)
- Never modify, move, rename or delete anything inside `Training/`, `Validation/` or `Testing/`.
- Never change the model architecture unless the task says so.
- Never edit files unrelated to the task. No reformatting, renaming or drive-by cleanups. Smallest possible diff.
- Never invent, round or estimate a metric. Numbers come from command output saved under `results/`. Write [NOT MEASURED] for anything missing.
- No secrets, credentials or personal data in files or logs.
- If a command fails twice for the same reason, stop and report the error instead of working around it.

## Engineering rules
- No hardcoded absolute paths; relative paths and argparse. No code that runs on import (`if __name__ == "__main__":`).
- Class names come from `labels.json` only. No hardcoded class lists or silent fallbacks.
- Models saved as `.keras`. Seeds fixed to 42 (python, numpy, tensorflow).
- Evaluation always loads the SAVED model file that the webcam and API use. Never retrain inside evaluation.
- Never remove duplicates from training data; exclude duplicates from test through a manifest file only.
- Don't swallow errors with bare `except: pass`.
- Use respectful terminology: "deaf and hard-of-hearing".

## Report format (end of every task)
Files changed, commands run, real outputs (trimmed), commit hash, and anything unresolved.

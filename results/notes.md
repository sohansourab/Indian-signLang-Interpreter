# Task 0 notes

## 1. Does `model.py` validate on the `Validation/` folder or on `validation_split` from `Training/`?

**It uses `validation_split=0.1` from whatever `--data-dir` points at (now `Training/`).**
It never reads the `Validation/` folder.

- `model.py`: `model.fit(X, y, ..., validation_split=0.1, callbacks=[lr_scheduler])`
- Keras takes the **last 10%** of the (shuffled) data passed to `fit`, so validation is a
  random in-memory slice of `Training/`, not the on-disk `Validation/` folder.
- Consequence: the `Validation/` folder is only used by `evaluate.py`, never during training.

## 2. Which script and command produced the earlier 98.12% test / 92.83% validation numbers?

**An uncommitted, ad-hoc temporary script — not the repository's own code.**

- Those two numbers came from a one-off evaluation script that loaded the previous
  **16-class** `ISL_Interpreter.keras` and ran it over the **`Testing/`** and
  **`Validation/`** folders, keeping only the 16 classes that were present in the old
  `labels.json` (A–P). That is why the effective test size was 3,198 of 5,200 images and
  the effective validation size was 1,576 of 2,579.
- No committed file produced them. The old `evaluate.py` **trained a fresh model** and
  applied `train_test_split(test_size=0.2, ...)` to a single folder; it never loaded a
  saved model and never evaluated on `Testing/` or `Validation/`.

## 3. Are those numbers reproducible from committed code?

**No.** At the time they were produced, the committed code could not reproduce them:
- the old `model.py`/`evaluate.py` had hardcoded A–P lists and a silent fallback, and
- the old `evaluate.py` retrained instead of evaluating the saved model.

After this task the pipeline is reproducible: `model.py` trains and saves the model plus
`results/history.json` and `results/training_curves.png`; `evaluate.py` loads the saved
model and evaluates it on `Testing/` and `Validation/`, writing `results/metrics.json`
and `results/confusion_matrix.png`.

## 4. Class coverage change

- Dataset folders: 26 classes (A–Z).
- Previously: `labels.json`, `model.py`'s `DEFAULT_LABELS`, and the saved model were all
  16 classes (A–P); `ISL_Interpreter.py` had a different 23-letter hardcoded fallback.
- Now: `labels.json` is regenerated from the 26 `Training/` folder names, every hardcoded
  list and silent fallback is removed, and the final `Dense` layer takes `len(labels)`
  (= 26). Training, evaluation, the webcam script and the FastAPI app all read the same
  `labels.json`.

## 5. Caveat carried into Task 1

The provided `Training/` and `Testing/` folders are known to share byte-identical images
(found in the earlier 16-class audit). The `Testing` numbers in `results/metrics.json`
therefore include those duplicates and are labelled as such. A duplicate manifest and a
clean-test evaluation are produced in Task 1.

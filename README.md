to display real-time data such as accuracy and loss for both the training data and the testing data and gives a graphical representation for the same over a number of trials of the model.
# Indian Sign Language Interpreter

A CNN-based fingerspelling recognizer that predicts hand-sign letters from images or a webcam. The project includes a local webcam application and a FastAPI browser interface.

This repository contains the source code and configuration only. The image dataset, virtual environment, and trained model are intentionally excluded from Git because of their size.

## Features

- Grayscale image preprocessing at 128 x 128 pixels
- Label definitions loaded from `labels.json`
- CNN training with Keras
- Local webcam prediction with a confidence threshold
- Browser-based prediction through FastAPI
- Docker support

## Requirements

- Python 3.12
- A webcam for live prediction
- TensorFlow, OpenCV, and the packages in `requirements.txt`

Python 3.12 is recommended because the project uses TensorFlow, which may not yet support every newer Python release.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset must use one directory per label, for example:

```text
data/
	A/
		image1.jpg
	B/
		image1.jpg
```

Keep the dataset outside the repository or in an ignored directory.

## Train the model

```powershell
python model.py --data-dir data --labels-path labels.json --model-path ISL_Interpreter.keras
```

The exact available options can be checked with:

```powershell
python model.py --help
```

Training metrics depend on the dataset and run configuration. No accuracy value is claimed here without a recorded evaluation result.

## Use the webcam

After a model has been trained:

```powershell
python ISL_Interpreter.py --model-path ISL_Interpreter.keras --labels-path labels.json
```

Press `q` in the camera window to stop the application.

## Run the web application

```powershell
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser and allow camera access when prompted.

The application expects `ISL_Interpreter.keras` in the project directory unless a different model path is configured in the application.

## Docker

Build and run the application with:

```powershell
docker build -t isl-interpreter .
docker run --rm -p 8000:8000 isl-interpreter
```

Browser camera access and model availability depend on the hosting environment. The model file is not included in this repository.

## Project files

- `model.py`: train and save the CNN model
- `ISL_Interpreter.py`: local webcam inference
- `web_app.py`: FastAPI backend
- `templates/index.html`: browser interface
- `labels.json`: label source of truth
- `evaluate.py`: evaluation workflow
- `Dockerfile`: container configuration

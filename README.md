# ASL Detection API

This project is an American Sign Language (ASL) letter detection system using a deep learning model served via a FastAPI backend and a simple frontend for image upload and prediction display.

## Features

- Train a custom CNN model for ASL letter classification using TensorFlow/Keras.
- Serve the trained model with FastAPI backend.
- Frontend interface to upload hand sign images and get predictions.
- API endpoints for prediction and health check.
- CORS enabled for frontend-backend communication.
- Model accuracy testing script included.

## Project Structure

- `train_model.py`: Script to train the ASL classification model.
- `main.py`: FastAPI backend serving the model and API endpoints.
- `index.html`: Frontend UI for uploading images and displaying predictions.
- `test_accuracy.py`: Script to evaluate the trained model on the test dataset.
- `requirements.txt`: Python dependencies.
- `asl_model.h5`: Trained model file (generated after training).
- `dataset/`: Contains training and test images organized by ASL letters.
- `asl_env/`: Python virtual environment directory.

## Setup Instructions

1. **Create and activate virtual environment**

```bash
python3 -m venv asl_env
source asl_env/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model**

```bash
python train_model.py
```

4. **Run the backend server**

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

5. **Open the frontend**

Open `index.html` in a browser or navigate to `http://127.0.0.1:8000` to use the web interface.

## API Endpoints

- `GET /`: Serves the frontend HTML page.
- `POST /api/predict`: Accepts an image file and returns the predicted ASL letter with confidence.
- `GET /health`: Returns the health status of the backend and model loading status.

## Testing Model Accuracy

Run the accuracy test script to evaluate the model on the test dataset:

```bash
python test_accuracy.py
```

## Notes

- The model input size is 64x64 RGB images.
- The frontend uploads images and sends them to the backend for prediction.
- The backend preprocesses images to match the model input format.
- The model is a custom CNN trained on the ASL dataset.

## Troubleshooting

- Ensure the virtual environment is activated before running scripts.
- If the model file `asl_model.h5` is missing, run the training script first.
- For any dependency issues, verify `requirements.txt` and reinstall packages.

## License

This project is open source and free to use.

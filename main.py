from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

app = FastAPI(title="ASL Detection API", description="API for American Sign Language letter detection")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ASL class labels (matching the dataset structure)
CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T",
    "U", "V", "W", "X", "Y"
]

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    model_path = "asl_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    else:
        print("Model not found. Please train the model first.")

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to 64x64 (matching model input)
        image = image.resize((64, 64))

        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0

        # Expand dims for batch
        image_array = np.expand_dims(image_array, axis=0)

        # Ensure shape is (1, 64, 64, 3)
        if image_array.shape != (1, 64, 64, 3):
            raise ValueError(f"Unexpected image shape: {image_array.shape}")

        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML file"""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Predict ASL letter from uploaded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read file content
        contents = await file.read()

        # Preprocess image
        processed_image = preprocess_image(contents)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        # Get predicted label
        predicted_label = CLASS_LABELS[predicted_class_idx]

        return {
            "label": predicted_label,
            "confidence": confidence,
            "class_index": int(predicted_class_idx)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

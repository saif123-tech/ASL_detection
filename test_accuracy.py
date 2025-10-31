import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
IMG_SIZE = 64
TEST_DIR = 'dataset/test'

# ASL class labels (matching the dataset structure)
CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T",
    "U", "V", "W", "X", "Y"
]

def test_model_accuracy():
    """Test the trained model on the test dataset"""
    # Load the model
    model_path = "asl_model.h5"
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical',
        classes=CLASS_LABELS,
        shuffle=False  # Important for accuracy calculation
    )

    print(f"Test samples: {test_generator.samples}")

    # Evaluate the model
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Calculate per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_LABELS))

    # Test a few sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(predicted_classes))):
        true_label = CLASS_LABELS[true_classes[i]]
        pred_label = CLASS_LABELS[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]]
        print(f"Sample {i+1}: True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    test_model_accuracy()

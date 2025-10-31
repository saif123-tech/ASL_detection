import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset paths
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'

# ASL class labels (matching the dataset structure)
CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T",
    "U", "V", "W", "X", "Y"
]

def create_model():
    """Create a custom CNN model for ASL classification"""
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_LABELS), activation='softmax')
    ])

    return model

def create_data_generators():
    """Create data generators for training and validation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_LABELS
    )

    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_LABELS
    )

    return train_generator, validation_generator

def train_model():
    """Train the ASL classification model"""
    print("Creating model...")
    model = create_model()

    print("Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators()

    print(f"Training data: {train_generator.samples} samples")
    print(f"Validation data: {validation_generator.samples} samples")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        'asl_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("Training completed!")

    # Evaluate on test set
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save final model
    model.save('asl_model.h5')
    print("Model saved as 'asl_model.h5'")

    # Check model size
    model_size = os.path.getsize('asl_model.h5') / (1024 * 1024)  # Size in MB
    print(f"Model size: {model_size:.2f} MB")

    if model_size > 25:
        print("Warning: Model size exceeds 25 MB. Consider reducing model complexity.")

    return model, history

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Train the model
    model, history = train_model()

    print("Training completed successfully!")


import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "waste_image_dataset")

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# Check dataset folders
for folder in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# Load validation images
val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Load test images
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_data.class_names
print("Detected classes:", class_names)

if len(class_names) < 2:
    raise ValueError("At least two category folders are required.")

if val_data.class_names != class_names:
    raise ValueError("Train and val categories do not match.")

if test_data.class_names != class_names:
    raise ValueError("Train and test categories do not match.")

# Improve input pipeline performance
train_data = train_data.prefetch(tf.data.AUTOTUNE)
val_data = val_data.prefetch(tf.data.AUTOTUNE)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

# Data augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
])

# Pretrained image model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Classification model
inputs = layers.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Evaluate on separate test dataset
test_loss, test_accuracy = model.evaluate(test_data)
print("Test accuracy:", round(test_accuracy, 4))

# Save model and class names
os.makedirs("models", exist_ok=True)

model.save("models/waste_classifier.keras")

with open("models/class_names.json", "w") as file:
    json.dump(class_names, file)

print("Training completed!")
print("Model saved in models/waste_classifier.keras")

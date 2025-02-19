import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import os
import pathlib
import shutil
from util.display_image import display_images
from util.visulizer import visualize_dataset


# Set up base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, 'datasets')

# Clean up existing dataset if it exists
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
    print("Cleaned up existing dataset directory")

# Create fresh dataset directory
os.makedirs(dataset_dir, exist_ok=True)

# Download and extract dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    origin=dataset_url,
    fname='flower_photos',
    untar=True,
)

# Convert to proper Path object and resolve any symlinks
data_dir = pathlib.Path(data_dir).resolve()

# Verify the correct path is being used
print(f"Using data directory: {data_dir}")

# Ensure the directory exists
if not data_dir.exists():
    raise RuntimeError(f"Data directory not found: {data_dir}")

# Count images
image_count = len(list(data_dir.rglob('*.jpg')))
print(f"Total images found: {image_count}")

# Dataset parameters
batch_size = 32
img_height = 180
img_width = 180

# Create training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_dir),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names
print("Available classes:", class_names)

# Show dataset info
print("\nDataset information:")
print(f"Number of training batches: {len(train_ds)}")
print(f"Number of validation batches: {len(val_ds)}")

visualize_dataset(train_ds, class_names, num_images=9, figsize=(10, 10))

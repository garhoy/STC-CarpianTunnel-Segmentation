import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
from metrics import compute_metrics_wrapper  # Custom metrics function (ignores background)
from utils import data_collator, MedicalSegmentationDataset
import matplotlib.pyplot as plt


"""
This script performs inference using a fine-tuned SegFormer model on a medical image segmentation task.

It computes pixel-wise semantic segmentation metrics (IoU, Precision, Recall, F1) using:
- Custom metric computation (Hand-coded)
- Optionally: Scikit-learn-based metric computation (commented here for simplicity)
All mean metrics reported exclude the background class (class 0).
"""


# Define class color palette: background + 4 classes
REFERENCE_COLORS = np.array([
    [0, 0, 0],        # class 0 → background
    [0, 255, 255],    # class 1 → nerve
    [255, 0, 255],    # class 2 → nerve edge
    [255, 128, 114],  # class 3 → ligament
    [255, 255, 0],    # class 4 → semilunar bone
], dtype=np.uint8)

# Paths to the test CSV, images and masks
VAL_CSV_PATH = "Data/Test_data.csv"
VAL_IMG_DIR  = "Data/test"
VAL_MASK_DIR = "Data/mask_test"

# Load the trained model and processor
model_path = "segformer-medical-output/final_model_noVal_focal"
processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Create inference dataset
val_dataset = MedicalSegmentationDataset(
    csv_file=VAL_CSV_PATH,
    image_dir=VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    processor=processor
)

# Inspect dataset
df = val_dataset.data
print(df.head())
print("Image directory:", val_dataset.image_dir)
print("Mask directory:", val_dataset.mask_dir)
print("Processor:", val_dataset.processor)

# Define inference (evaluation) arguments
eval_args = TrainingArguments(
    output_dir="./segformer-medical-output",
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50
)

# Create trainer with custom metric computation
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapper  # Excludes background class from mean metrics
)

# Run inference
results = trainer.predict(val_dataset)

# Extract and print metrics
print("Validation metrics (excluding background):", results.metrics)
metrics_hand = results.metrics

# Convert to DataFrame for cleaner printing or saving
df = pd.DataFrame({
    'Hand-coded': metrics_hand,
})
df = df.round(6)

print(df)

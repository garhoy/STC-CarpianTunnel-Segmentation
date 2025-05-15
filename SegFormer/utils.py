import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import Trainer

"""
Custom PyTorch Dataset and Trainer for Semantic Segmentation with SegFormer.

Includes:
- `MedicalSegmentationDataset`: Loads images and masks from CSVs and applies preprocessing.
- `data_collator`: Batches input tensors and labels for training.
- `CustomTrainer`: Extension of HuggingFace's `Trainer` supporting Cross Entropy, Focal, and Dice loss.

Features:
- Pixel-wise label generation from RGB masks using a color palette.
- Loss interpolation to match prediction and ground truth resolution.
- Smooth one-hot encoding for dice loss computation.
- Easily switch between loss types with `loss_name` parameter.

"""

REFERENCE_COLORS = np.array([
    [0,   0,   0],    # 0 background
    [0, 255, 255],    # 1 nerve
    [255, 0, 255],    # 2 nerve edge 
    [255,128,114],    # 3 ligament
    [255,255,  0],    # 4 Semilunar Bone
], dtype=np.uint8)

COLOR_THRESHOLD = 20  

def rgb_to_label(mask_rgb, reference_colors, threshold=20):

    H, W, _ = mask_rgb.shape
    label_map = np.zeros((H, W), dtype=np.uint8)
    mask_rgb_int = mask_rgb.astype(np.int16)
    
    for idx, color in enumerate(reference_colors):
        color_int = color.astype(np.int16)
        match = np.all(np.abs(mask_rgb_int - color_int) <= threshold, axis=-1)
        label_map[match] = idx
    return label_map

# =============================================================================
# Dataset for Medical Segmentation 
# =============================================================================

class MedicalSegmentationDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, processor):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx]["image"]
        mask_filename = self.data.iloc[idx]["mask"]

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            mask_np = rgb_to_label(mask_np, REFERENCE_COLORS, threshold=COLOR_THRESHOLD)
        else:
            mask_np = np.array(mask).astype("long")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  

        return {"pixel_values": pixel_values, "labels": torch.tensor(mask_np, dtype=torch.long)}
    

# =============================================================================
# Puts examples in batches
# =============================================================================

def data_collator(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# =============================================================================
# Custom Trainer Definition
# =============================================================================

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_name="cross_entropy", **kwargs):
        """
        loss_name: 
            "cross_entropy" , "focal" o "dice".
        """
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Logits need to have same shape or size as masks.
        """
        labels = inputs.get("labels")  
        outputs = model(**inputs)
        logits = outputs.logits 

        if logits.shape[2:] != labels.shape[1:]:
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[1:], mode="nearest"
            )

        if self.loss_name in ["cross_entropy", "focal"]:
            logits_reshaped = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
            labels_reshaped = labels.view(-1)
            
            if self.loss_name == "cross_entropy":
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits_reshaped, labels_reshaped)
            elif self.loss_name == "focal":
                gamma = 2.0
                alpha = 0.25
                ce_loss = torch.nn.functional.cross_entropy(
                    logits_reshaped, labels_reshaped, reduction="none"
                )
                pt = torch.exp(-ce_loss)
                focal_loss = alpha * (1 - pt) ** gamma * ce_loss
                loss = focal_loss.mean()

        elif self.loss_name == "dice":
            num_classes = logits.shape[1]
            smooth = 1e-6
            probs = torch.softmax(logits, dim=1)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
            labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()

            dice_loss = 0.0
            count = 0
            for cls in range(1, num_classes):
                prob_cls = probs[:, cls, :, :]
                target_cls = labels_one_hot[:, cls, :, :]
                intersection = (prob_cls * target_cls).sum(dim=(1, 2))
                union = prob_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
                dice_score = (2 * intersection + smooth) / (union + smooth)
                dice_loss_cls = 1 - dice_score
                dice_loss += dice_loss_cls.mean()
                count += 1
            loss = dice_loss / count if count > 0 else dice_loss

        else:
            raise ValueError(f"Unknown loss_name: {self.loss_name}")

        return (loss, outputs) if return_outputs else loss





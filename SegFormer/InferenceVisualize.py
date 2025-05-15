import os
import pandas as pd
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from utils import rgb_to_label, REFERENCE_COLORS, COLOR_THRESHOLD

'''
Quick visualization script to display SegFormer predictions vs ground truth
on a medical segmentation dataset (e.g., carpal tunnel ultrasound).

For each image:
- Runs model inference
- Computes mIoU
- Shows image with predicted and GT overlays
'''
def compute_iou_per_class(pred_mask, gt_mask, num_classes):

    ious = {}
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        gt_inds = (gt_mask == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        if union == 0:
            ious[cls] = float('nan')
        else:
            ious[cls] = intersection / union
    return ious


def compute_mean_iou(pred_mask, gt_mask, num_classes):
    ious = compute_iou_per_class(pred_mask, gt_mask, num_classes)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    if len(valid_ious) == 0:
        return float('nan')
    return np.mean(valid_ious)

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
        mask_filename  = self.data.iloc[idx]["mask"]

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path  = os.path.join(self.mask_dir,  mask_filename)

        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            mask_np = rgb_to_label(mask_np, REFERENCE_COLORS, threshold=COLOR_THRESHOLD)
        else:
            mask_np = mask_np.astype("long")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels":       torch.tensor(mask_np, dtype=torch.long),
            "filename":     image_filename,
            "img_path":     image_path,
        }


####################################################
#          Paths and Model Configuration           #
####################################################

model_path    = "segformer-medical-output/final_model_noVal_focal"
processor     = SegformerImageProcessor.from_pretrained(model_path)
model         = SegformerForSemanticSegmentation.from_pretrained(model_path)
device        = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

VAL_CSV_PATH = "Data/Test_data.csv"
VAL_IMG_DIR  = "Data/test"
VAL_MASK_DIR = "Data/mask_test"

val_dataset = MedicalSegmentationDataset(
    csv_file=VAL_CSV_PATH,
    image_dir=VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    processor=processor
)


def overlay_segmentation(idx, num_classes=len(REFERENCE_COLORS)):
    sample = val_dataset[idx]
    img_pil = Image.open(sample["img_path"]).convert("RGB")
    img = np.array(img_pil)

    input_tensor = sample["pixel_values"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor).logits
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    gt = sample["labels"].cpu().numpy()

    # Upsample if different size.
    H0, W0 = img.shape[:2]
    pred_up = cv2.resize(pred.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
    gt_up   = cv2.resize(gt.astype(np.uint8),   (W0, H0), interpolation=cv2.INTER_NEAREST)

    miou = compute_mean_iou(pred_up, gt_up, num_classes)
    print(f"Image: {sample['filename']} - mIoU: {miou:.4f}")

    def make_overlay(mask):
        overlay = REFERENCE_COLORS[mask]
        alpha   = 0.5
        return np.dstack([overlay, np.ones((H0, W0)) * alpha])

    ov_pred = make_overlay(pred_up)
    ov_gt   = make_overlay(gt_up)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img); axes[0].imshow(ov_pred)
    axes[0].set_title(f"Prediction\nmIoU: {miou:.4f}"); axes[0].axis("off")
    axes[1].imshow(img); axes[1].imshow(ov_gt)
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for i in range(10):
        overlay_segmentation(i)



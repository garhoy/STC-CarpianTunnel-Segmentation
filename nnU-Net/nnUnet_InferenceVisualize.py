import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
This script visualizes nnU-Net predictions on test cases by overlaying
the predicted and ground truth segmentation masks onto the original ultrasound image.
It also computes and displays the mean Intersection over Union (mIoU) per case.

Requirements:
- NIfTI files for test images, ground truth masks, and predicted masks.
- Color palette defining semantic labels.

Output:
- Side-by-side visual comparison of predictions and ground truth with mIoU score.
"""


# === Paths ===
IMG_DIR        = "nnUnetDataFormat/nnUNet_raw_data_base/nnUNet_raw_data/Task000_STC/imagesTs"
GT_MASK_DIR    = "nnUnetDataFormat/nnUNet_raw_data_base/nnUNet_raw_data/Task000_STC/labelsTs"
PRED_MASK_DIR  = "nnUnetDataFormat/predictions"

# === Color palette (class 0 to 4) ===
REFERENCE_COLORS = np.array([
    [0,   0,   0],    # 0: background
    [0, 255, 255],    # 1: nerve
    [255, 0, 255],    # 2: nerve border
    [255,128,114],    # 3: ligament
    [255,255,  0],    # 4: semilunar bone
], dtype=np.uint8)

def colorize_mask(mask):
    """Apply RGB color to each class label using the reference palette."""
    return REFERENCE_COLORS[mask]

def compute_iou(pred, gt, num_classes=5):
    """
    Compute mean IoU across all classes (including background).
    If you want to exclude background, change range to range(1, num_classes).
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c   = (gt == c)
        if np.logical_or(pred_c, gt_c).sum() == 0:
            continue
        iou = np.logical_and(pred_c, gt_c).sum() / np.logical_or(pred_c, gt_c).sum()
        ious.append(iou)
    return np.mean(ious) if ious else np.nan

def overlay_segmentation(case_id):
    """
    Show overlay comparison between prediction and ground truth for a given case ID.
    """
    pred_path = os.path.join(PRED_MASK_DIR, f"{case_id}.nii.gz")
    gt_path   = os.path.join(GT_MASK_DIR,   f"{case_id}.nii.gz")
    img_path  = os.path.join(IMG_DIR,       f"{case_id}_0000.nii.gz")

    if not all(map(os.path.exists, [pred_path, gt_path, img_path])):
        print(f" Missing files for {case_id}")
        return

    image     = nib.load(img_path).get_fdata().squeeze()
    pred_mask = nib.load(pred_path).get_fdata().astype(np.uint8).squeeze()
    gt_mask   = nib.load(gt_path).get_fdata().astype(np.uint8).squeeze()

    # Normalize image to 0–255 and convert to RGB
    image = np.clip(image, 0, np.percentile(image, 99))
    image = (255 * (image - image.min()) / (image.ptp() + 1e-5)).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize masks to image shape (in case of any mismatch)
    H, W = image.shape[:2]
    pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    gt_mask   = cv2.resize(gt_mask,   (W, H), interpolation=cv2.INTER_NEAREST)

    pred_overlay = colorize_mask(pred_mask)
    gt_overlay   = colorize_mask(gt_mask)

    # Compute mean IoU (including background)
    miou = compute_iou(pred_mask, gt_mask, num_classes=5)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image); axes[0].imshow(pred_overlay, alpha=0.5)
    axes[0].set_title(f"Prediction\nmIoU: {miou:.4f}"); axes[0].axis("off")
    axes[1].imshow(image); axes[1].imshow(gt_overlay, alpha=0.5)
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# === Loop through test cases and visualize ===
if __name__ == "__main__":
    for i in range(10):  # Adjust number of samples as needed
        case_id = f"TS_{i:04d}"
        overlay_segmentation(case_id)

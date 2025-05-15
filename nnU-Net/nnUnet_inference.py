import os
import time
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (
    jaccard_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)


"""
This script computes evaluation metrics for nnU-Net predictions on a test set.
It compares the predicted segmentation masks against the ground truth using
sklearn metrics such as IoU, Precision, Recall, F1-score, and Dice.
It also optionally computes Dice loss and Cross-Entropy loss using softmax probabilities (if available).
Metrics are aggregated per class and as global averages (excluding background).
"""

# Number of classes including background (0)
NUM_CLASSES = 5

# Paths to predictions, softmax outputs, and ground truth labels
PRED_LABEL_DIR = "nnUnetDataFormat/predictions"
PRED_PROB_DIR  = "nnUnetDataFormat/predictions_npz"
GT_DIR         = "nnUnetDataFormat/nnUNet_raw_data_base/nnUNet_raw_data/Task000_STC/labelsTs"

def compute_metrics_sklearn(vol_pred, vol_gt):
    """
    Compute per-class and average segmentation metrics using sklearn.
    Returns IoU, precision, recall, F1, and Dice (≈F1) for each class.
    Averages ignore background (class 0).
    """
    y_true = vol_gt.flatten()
    y_pred = vol_pred.flatten()
    classes = list(range(NUM_CLASSES))

    iou_arr       = jaccard_score(y_true, y_pred, average=None, labels=classes)
    precision_arr = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    recall_arr    = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_arr        = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    # Set NaN for classes missing in both prediction and ground truth
    for c in [c for c in classes if not (y_true == c).any() and not (y_pred == c).any()]:
        iou_arr[c] = precision_arr[c] = recall_arr[c] = f1_arr[c] = np.nan

    metrics = {}
    for c in classes:
        metrics[f"iou_class_{c}"]       = float(iou_arr[c])
        metrics[f"precision_class_{c}"] = float(precision_arr[c])
        metrics[f"recall_class_{c}"]    = float(recall_arr[c])
        metrics[f"f1_class_{c}"]        = float(f1_arr[c])
        metrics[f"dice_class_{c}"]      = float(f1_arr[c])  # Dice ≈ F1

    # Compute mean metrics excluding background
    valid = [c for c in classes if c != 0]
    metrics["mean_iou"]       = float(np.nanmean(iou_arr[valid]))
    metrics["mean_precision"] = float(np.nanmean(precision_arr[valid]))
    metrics["mean_recall"]    = float(np.nanmean(recall_arr[valid]))
    metrics["mean_f1"]        = float(np.nanmean(f1_arr[valid]))
    metrics["mean_dice"]      = metrics["mean_f1"]

    return metrics

def dice_loss_from_masks(vol_pred, vol_gt):
    """
    Compute average Dice loss across all classes.
    """
    y_true = vol_gt.flatten()
    y_pred = vol_pred.flatten()
    losses = []
    for c in range(NUM_CLASSES):
        mask_t = (y_true == c).astype(int)
        mask_p = (y_pred == c).astype(int)
        d = f1_score(mask_t, mask_p, zero_division=1)
        losses.append(1 - d)
    return float(np.mean(losses))

def ce_loss_from_npz(npz_path, vol_gt):
    """
    Compute cross-entropy loss from predicted softmax probabilities.
    """
    data = np.load(npz_path)
    probs = data.get("softmax", next(iter(data.values())))
    C = probs.shape[0]
    flat_probs = probs.reshape(C, -1).T
    flat_gt    = vol_gt.flatten()
    return float(log_loss(flat_gt, flat_probs, labels=list(range(NUM_CLASSES))))

def main():
    pred_files = sorted(glob(os.path.join(PRED_LABEL_DIR, "*.nii.gz")))
    rows = []

    start_time = time.time()
    for pred_f in tqdm(pred_files, desc="Processing cases", unit="case"):
        case_id = os.path.basename(pred_f).replace(".nii.gz", "")
        gt_f    = os.path.join(GT_DIR, os.path.basename(pred_f))
        npz_f   = os.path.join(PRED_PROB_DIR, case_id + ".npz")

        if not os.path.exists(gt_f):
            tqdm.write(f"  Ground truth missing for {case_id}, skipping.")
            continue

        pred_vol = nib.load(pred_f).get_fdata().astype(int)
        gt_vol   = nib.load(gt_f).get_fdata().astype(int)
        if pred_vol.shape != gt_vol.shape:
            tqdm.write(f" Shape mismatch: {case_id} -> pred {pred_vol.shape} vs gt {gt_vol.shape}")
            continue

        unique, counts = np.unique(gt_vol, return_counts=True)
        tqdm.write(f"{case_id} → GT voxel distribution: {dict(zip(unique.astype(int), counts))}")

        seg_m = compute_metrics_sklearn(pred_vol, gt_vol)
        dl    = dice_loss_from_masks(pred_vol, gt_vol)
        closs = ce_loss_from_npz(npz_f, gt_vol) if os.path.exists(npz_f) else np.nan

        row = {"case_id": case_id, **seg_m, "dice_loss": dl, "ce_loss": closs}
        rows.append(row)

    df = pd.DataFrame(rows).set_index("case_id")

    # Print global averages (excluding background)
    print("\n=== Global Averages (excluding background) ===")
    for key in ["mean_iou", "mean_dice", "mean_precision", "mean_recall", "mean_f1", "dice_loss", "ce_loss"]:
        if key in df:
            print(f"{key:>15} : {df[key].mean():.4f}")

    # Print per-class averages
    print("\n=== Per-Class Metrics (averaged across cases) ===")
    for c in range(NUM_CLASSES):
        print(
            f"Class {c}: "
            f"IoU={df[f'iou_class_{c}'].mean():.4f} | "
            f"Dice={df[f'dice_class_{c}'].mean():.4f} | "
            f"Prec={df[f'precision_class_{c}'].mean():.4f} | "
            f"Rec={df[f'recall_class_{c}'].mean():.4f} | "
            f"F1={df[f'f1_class_{c}'].mean():.4f}"
        )

    out_csv = "nnunet_full_metrics_sklearn.csv"
    # df.to_csv(out_csv)
    print(f"\n✅ Per-case metrics saved to '{out_csv}'.")

    total_time = time.time() - start_time
    print(f"\n Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

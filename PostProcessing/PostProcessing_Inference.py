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
from metrics import compute_metrics

'''
This Python script performs semantic segmentation inference using a pre-trained SegFormer model on medical images.
It applies post-processing to refine predictions for the semilunar bone (class 4),
ensuring only the largest connected component located below the nerve (class 1) is retained.
It visualizes the original prediction, the post-processed prediction, and the ground truth,
along with their respective mIoU, F1, Recall, and Precision scores per image.
Finally, it calculates and reports the aggregated mean metrics for the entire test set
for both original and post-processed predictions.
'''

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

# Define your REFERENCE_COLORS and COLOR_THRESHOLD as you have them
REFERENCE_COLORS = np.array([
    [0, 0, 0],        # class 0 → background
    [0, 255, 255],    # class 1 → nerve
    [255, 0, 255],    # class 2 → nerve edge
    [255, 128, 114],  # class 3 → ligament
    [255, 255, 0],    # class 4 → semilunar bone
], dtype=np.uint8)

COLOR_THRESHOLD = 20 # Example value, adjust as needed

####################################################
#          Paths and Model Configuration           #
####################################################

model_path    = "segformer-medical-output/final_model_noVal_cross"
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

def post_process_semilunar_bone(pred_mask, nerve_class=1, semilunar_class=4):
    """
    Applies post-processing to consolidate the semilunar bone prediction.
    - Identifies connected components for the semilunar bone.
    - Selects the largest component that is below the nerve.

    Args:
        pred_mask (np.array): The predicted segmentation mask.
        nerve_class (int): The label for the nerve class.
        semilunar_class (int): The label for the semilunar bone class.

    Returns:
        np.array: The post-processed prediction mask.
    """
    processed_mask = pred_mask.copy()
    H, W = pred_mask.shape

    # 1. Isolate the semilunar bone prediction
    semilunar_mask = (pred_mask == semilunar_class).astype(np.uint8)

    # 2. Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        semilunar_mask, connectivity=8
    )

    # Ignore background label (label 0)
    candidate_labels = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 25:  # filter out components smaller than 10 pixels
            candidate_labels.append(i)

    if not candidate_labels:
        return processed_mask

    # 3. Determine nerve's vertical position for spatial filtering
    nerve_pixels = np.where(pred_mask == nerve_class)
    if nerve_pixels[0].size == 0:
        best_label = -1
        max_area = -1
        for label_idx in candidate_labels:
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                best_label = label_idx
        if best_label != -1:
            # Create a new mask with only the largest component
            final_semilunar_mask = (labels == best_label) * semilunar_class
            processed_mask[processed_mask == semilunar_class] = 0 # Clear existing semilunar
            processed_mask = np.maximum(processed_mask, final_semilunar_mask) # Add the selected one
        return processed_mask


    # Get the approximate bottom y-coordinate of the nerve (or average/median)
    nerve_bottom_y = np.max(nerve_pixels[0]) if nerve_pixels[0].size > 0 else 0

    best_label = -1
    max_area_below_nerve = -1

    for label_idx in candidate_labels:
        area = stats[label_idx, cv2.CC_STAT_AREA]
        # Get the top-most y-coordinate or centroid y-coordinate of the component
        comp_top_y = stats[label_idx, cv2.CC_STAT_TOP]
        comp_centroid_y = centroids[label_idx, 1]

        # Condition: is the component *below* the nerve?
        if comp_centroid_y > nerve_bottom_y:
            if area > max_area_below_nerve:
                max_area_below_nerve = area
                best_label = label_idx

    # 5. Reconstruct the mask
    if best_label != -1:
        final_semilunar_mask = (labels == best_label) * semilunar_class

        processed_mask[processed_mask == semilunar_class] = 0
        processed_mask = np.maximum(processed_mask, final_semilunar_mask)
    else:
        processed_mask[processed_mask == semilunar_class] = 0

    return processed_mask

def overlay_segmentation_and_get_masks(idx, num_classes=len(REFERENCE_COLORS), plot_results=True):
    sample = val_dataset[idx]
    img_pil = Image.open(sample["img_path"]).convert("RGB")
    img = np.array(img_pil)

    input_tensor = sample["pixel_values"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor).logits
        pred_logits = logits.cpu().numpy()
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    gt = sample["labels"].cpu().numpy()

    # Upsample if different size.
    H0, W0 = img.shape[:2]
    pred_up = cv2.resize(pred.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
    gt_up   = cv2.resize(gt.astype(np.uint8),   (W0, H0), interpolation=cv2.INTER_NEAREST)

    # --- Apply Post-Processing ---
    pred_up_postprocessed = post_process_semilunar_bone(pred_up.copy(), nerve_class=1, semilunar_class=4)
    # ---------------------------

    if plot_results:
        # Calculate metrics for original prediction (for display)
        eval_pred_original = (pred_logits, gt_up[np.newaxis, :, :])
        metrics_original = compute_metrics(eval_pred_original, num_classes=num_classes, ignore_index=0)

        # Calculate metrics for post-processed prediction (for display)
        # Create dummy logits for post-processed mask
        pred_up_postprocessed_one_hot = np.eye(num_classes)[pred_up_postprocessed]
        pred_up_postprocessed_logits = np.transpose(pred_up_postprocessed_one_hot, (2, 0, 1))[np.newaxis, :, :, :]

        eval_pred_postprocessed = (pred_up_postprocessed_logits, gt_up[np.newaxis, :, :])
        metrics_postprocessed = compute_metrics(eval_pred_postprocessed, num_classes=num_classes, ignore_index=0)

        print(f"--- Image: {sample['filename']} ---")
        print(f"Original Prediction Metrics (No Background):")
        print(f"  mIoU: {metrics_original['mean_iou_no_bg']:.4f}")
        print(f"  mF1:  {metrics_original['mean_f1_no_bg']:.4f}")
        print(f"  mRecall: {metrics_original['mean_recall_no_bg']:.4f}")
        print(f"  mPrecision: {metrics_original['mean_precision_no_bg']:.4f}")

        print(f"Post-Processed Prediction Metrics (No Background):")
        print(f"  mIoU: {metrics_postprocessed['mean_iou_no_bg']:.4f}")
        print(f"  mF1:  {metrics_postprocessed['mean_f1_no_bg']:.4f}")
        print(f"  mRecall: {metrics_postprocessed['mean_recall_no_bg']:.4f}")
        print(f"  mPrecision: {metrics_postprocessed['mean_precision_no_bg']:.4f}")

        def make_overlay(mask):
            overlay = REFERENCE_COLORS[mask]
            alpha   = 0.5
            return np.dstack([overlay, np.ones((H0, W0)) * alpha])

        ov_pred_original = make_overlay(pred_up)
        ov_pred_postprocessed = make_overlay(pred_up_postprocessed)
        ov_gt   = make_overlay(gt_up)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img); axes[0].imshow(ov_pred_original)
        axes[0].set_title(f"Original Prediction\nmIoU: {metrics_original['mean_iou_no_bg']:.4f}"); axes[0].axis("off")

        axes[1].imshow(img); axes[1].imshow(ov_pred_postprocessed)
        axes[1].set_title(f"Post-Processed Prediction\nmIoU: {metrics_postprocessed['mean_iou_no_bg']:.4f}"); axes[1].axis("off")

        axes[2].imshow(img); axes[2].imshow(ov_gt)
        axes[2].set_title("Ground Truth"); axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    # Return the upsampled predictions and ground truth for aggregate calculation
    return pred_up, pred_up_postprocessed, gt_up, pred_logits # Return logits for original

if __name__ == "__main__":
    all_original_preds_logits = []
    all_postprocessed_preds_masks = [] # We'll store masks and create dummy logits later
    all_gts = []

    # Iterate through all images in the validation dataset
    # Change range(10) to len(val_dataset) to process the entire test set
    for i in range(len(val_dataset)):
        # Set plot_results to False if you don't want individual image plots for all images
        # For a large test set, you might only plot the first few or none
        pred_original_mask, pred_postprocessed_mask, gt_mask, original_logits = \
            overlay_segmentation_and_get_masks(i, plot_results=(i < 5)) # Plot only first 5 as example

        # Append data for aggregate calculation
        all_original_preds_logits.append(original_logits)
        all_postprocessed_preds_masks.append(pred_postprocessed_mask)
        all_gts.append(gt_mask)

    # --- Aggregate Metrics Calculation ---
    print("\n" + "="*50)
    print("      AGGREGATE METRICS FOR THE ENTIRE TEST SET      ")
    print("="*50)

    num_classes = len(REFERENCE_COLORS)

    # Prepare data for original predictions
    # Stack logits for original predictions along a new batch dimension
    stacked_original_logits = np.concatenate(all_original_preds_logits, axis=0) # Should be [N, C, H, W]
    stacked_gts_original = np.stack(all_gts, axis=0) # Should be [N, H, W]

    overall_eval_pred_original = (stacked_original_logits, stacked_gts_original)
    overall_metrics_original = compute_metrics(overall_eval_pred_original, num_classes=num_classes, ignore_index=0)

    print("\n--- Overall Original Prediction Metrics (No Background) ---")
    print(f"  Mean IoU:       {overall_metrics_original['mean_iou_no_bg']:.4f}")
    print(f"  Mean F1 Score:  {overall_metrics_original['mean_f1_no_bg']:.4f}")
    print(f"  Mean Recall:    {overall_metrics_original['mean_recall_no_bg']:.4f}")
    print(f"  Mean Precision: {overall_metrics_original['mean_precision_no_bg']:.4f}")
    print("--- Per-Class Metrics (Original) ---")
    for cls in range(num_classes):
        if cls == 0: # Skip background class
            continue
        print(f"  Class {cls} (IoU/F1/Recall/Precision): "
              f"{overall_metrics_original[f'iou_class_{cls}']:.4f} / "
              f"{overall_metrics_original[f'f1_class_{cls}']:.4f} / "
              f"{overall_metrics_original[f'recall_class_{cls}']:.4f} / "
              f"{overall_metrics_original[f'precision_class_{cls}']:.4f}")


    # Prepare data for post-processed predictions
    # Convert stacked masks to 'dummy logits' for compute_metrics
    stacked_postprocessed_masks = np.stack(all_postprocessed_preds_masks, axis=0)
    # Create one-hot encoded 'logits' from the post-processed masks
    stacked_postprocessed_one_hot = np.eye(num_classes)[stacked_postprocessed_masks]
    # Transpose to [N, C, H, W]
    stacked_postprocessed_logits = np.transpose(stacked_postprocessed_one_hot, (0, 3, 1, 2)) # N, H, W, C -> N, C, H, W

    stacked_gts_postprocessed = np.stack(all_gts, axis=0) # Same GTs

    overall_eval_pred_postprocessed = (stacked_postprocessed_logits, stacked_gts_postprocessed)
    overall_metrics_postprocessed = compute_metrics(overall_eval_pred_postprocessed, num_classes=num_classes, ignore_index=0)

    print("\n--- Overall Post-Processed Prediction Metrics (No Background) ---")
    print(f"  Mean IoU:       {overall_metrics_postprocessed['mean_iou_no_bg']:.4f}")
    print(f"  Mean F1 Score:  {overall_metrics_postprocessed['mean_f1_no_bg']:.4f}")
    print(f"  Mean Recall:    {overall_metrics_postprocessed['mean_recall_no_bg']:.4f}")
    print(f"  Mean Precision: {overall_metrics_postprocessed['mean_precision_no_bg']:.4f}")
    print("--- Per-Class Metrics (Post-Processed) ---")
    for cls in range(num_classes):
        if cls == 0: # Skip background class
            continue
        print(f"  Class {cls} (IoU/F1/Recall/Precision): "
              f"{overall_metrics_postprocessed[f'iou_class_{cls}']:.4f} / "
              f"{overall_metrics_postprocessed[f'f1_class_{cls}']:.4f} / "
              f"{overall_metrics_postprocessed[f'recall_class_{cls}']:.4f} / "
              f"{overall_metrics_postprocessed[f'precision_class_{cls}']:.4f}")
import os
import pandas as pd
import numpy as np
import nibabel as nib
from PIL import Image
import json
"""
This script prepares a 2D medical image dataset for nnU-Net training and evaluation.
It performs the following steps:

1. Reads image/mask paths from three CSV files: Train, Val, and Test.
2. Merges Train and Val into a single training set, while keeping Test separate.
3. Converts images and masks from common formats (e.g. PNG) to NIfTI (.nii.gz).
   - For images: Converts to grayscale and adds channel dimension.
   - For masks: Matches RGB colors to class labels (0 to 4), with class 0 as background.
4. Saves images and labels in the nnU-Net expected folder structure:
   - imagesTr/, labelsTr/, imagesTs/, labelsTs/
5. Generates a compatible dataset.json describing the task, modalities, labels, and file mappings.

Usage:
- Set the appropriate file paths for CSVs and folders.
- Run the script to populate your nnU-Net raw data directory with formatted data.
"""

# =============================================================================
# Paths
# =============================================================================
TRAIN_CSV_PATH = "Data/Train_data.csv"
VAL_CSV_PATH   = "Data/Val_Data.csv"
TEST_CSV_PATH  = "Data/Test_data.csv"

TRAIN_IMG_DIR  = "Data/train"
TRAIN_MASK_DIR = "Data/mask_train"

VAL_IMG_DIR    = "Data/val"
VAL_MASK_DIR   = "Data/mask_val"

TEST_IMG_DIR   = "Data/test"
TEST_MASK_DIR  = "Data/mask_test"

# =============================================================================
# Directory for  NNU-NET
# =============================================================================
TASK_NAME     = "Task000_STC"
BASE_TASK_DIR = ("nnUnetDataFormat/"
                 "nnUNet_raw_data_base/nnUNet_raw_data/" + TASK_NAME)

IMAGES_TR_DIR = os.path.join(BASE_TASK_DIR, "imagesTr")
LABELS_TR_DIR = os.path.join(BASE_TASK_DIR, "labelsTr")
IMAGES_TS_DIR = os.path.join(BASE_TASK_DIR, "imagesTs")
LABELS_TS_DIR = os.path.join(BASE_TASK_DIR, "labelsTs")

for d in [IMAGES_TR_DIR, LABELS_TR_DIR, IMAGES_TS_DIR, LABELS_TS_DIR]:
    os.makedirs(d, exist_ok=True)


# Define class color palette: background + 4 classes
REFERENCE_COLORS = np.array([
    [0, 0, 0],        # class 0 → background
    [0, 255, 255],    # class 1 → nerve
    [255, 0, 255],    # class 2 → nerve edge
    [255, 128, 114],  # class 3 → ligament
    [255, 255, 0],    # class 4 → semilunar bone
], dtype=np.uint8)


def convert_to_nii(inp_path, out_path, is_mask=False):
    img = Image.open(inp_path)
    if is_mask:
        arr = np.array(img.convert("RGB"))
        H,W,_ = arr.shape
        lbl = np.zeros((H,W), dtype=np.uint8)               # background=0
        for idx, color in enumerate(REFERENCE_COLORS[1:], 1):# Only class 1 to 4 
            mask = np.all(arr == color, axis=-1)            # exact match
            lbl[mask] = idx
        data = lbl[...,None]
    else:
        grey = np.array(img.convert("L"))[...,None]
        data = grey

    nii = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(nii, out_path)


# =============================================================================
#  Mix up Train and Val 
# =============================================================================
def process_split(csv_path, img_dir, msk_dir, out_img_dir, out_lbl_dir,
                  prefix, start_idx=0, to_json=True):
    df = pd.read_csv(csv_path)
    json_entries = []
    for i, row in df.iterrows():
        idx = start_idx + i
        case = f"{prefix}_{idx:04d}"
        img_in = os.path.join(img_dir, row["image"])
        msk_in = os.path.join(msk_dir, row["mask"])
        img_out = os.path.join(out_img_dir, f"{case}_0000.nii.gz")
        lbl_out = os.path.join(out_lbl_dir, f"{case}.nii.gz")
        convert_to_nii(img_in, img_out, is_mask=False)
        convert_to_nii(msk_in, lbl_out, is_mask=True)
        if to_json:
            json_entries.append({
                "image": f"./imagesTr/{case}.nii.gz",
                "label": f"./labelsTr/{case}.nii.gz"
            })
    return json_entries

print("→ Processing TRAIN...")
train_list = process_split(TRAIN_CSV_PATH,
                           TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                           IMAGES_TR_DIR, LABELS_TR_DIR,
                           prefix="TR", start_idx=0)

print("→ Processing VAL mixed with Train ...")
val_list = process_split(VAL_CSV_PATH,
                         VAL_IMG_DIR, VAL_MASK_DIR,
                         IMAGES_TR_DIR, LABELS_TR_DIR,
                         prefix="TR", start_idx=len(train_list))
train_list += val_list

# =============================================================================
# PROCESAR TEST por separado
# =============================================================================
def process_test(csv_path, img_dir, msk_dir, out_img_dir, out_lbl_dir,
                 prefix="TS", start_idx=0):
    df = pd.read_csv(csv_path)
    img_entries = []
    for i, row in df.iterrows():
        case = f"{prefix}_{i:04d}"
        img_in = os.path.join(img_dir, row["image"])
        msk_in = os.path.join(msk_dir, row["mask"])
        img_out = os.path.join(out_img_dir, f"{case}_0000.nii.gz")
        lbl_out = os.path.join(out_lbl_dir, f"{case}.nii.gz")
        convert_to_nii(img_in, img_out, is_mask=False)
        convert_to_nii(msk_in, lbl_out, is_mask=True)
        img_entries.append(f"./imagesTs/{case}.nii.gz")
    return img_entries

print("→ Processing TEST...")
test_list = process_test(TEST_CSV_PATH,
                         TEST_IMG_DIR, TEST_MASK_DIR,
                         IMAGES_TS_DIR, LABELS_TS_DIR)

# =============================================================================
# GENERATE dataset.json
# =============================================================================
ds = {
    "name": TASK_NAME,
    "description": "Ecografías túnel carpiano",
    "tensorImageSize": "2D",
    "modality": {"0":"US"},
    "labels": {
        "0":"background","1":"nervio","2":"BordeNervio",
        "3":"LigamentoDelCarpo","4":"Hueso Semilunar"
    },
    "numTraining": len(train_list),
    "numTest": len(test_list),
    "training": train_list,
    "test": test_list
}
with open(os.path.join(BASE_TASK_DIR,"dataset.json"), "w") as f:
    json.dump(ds, f, indent=4)
print(f"✅ dataset.json: {len(train_list)} train, {len(test_list)} test")

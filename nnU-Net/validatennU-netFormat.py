import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

"""
This script visualizes NIfTI (.nii.gz) medical images and their corresponding segmentation masks.
It displays:
  1. The original grayscale ultrasound image.
  2. The segmentation mask (grayscale).
  3. An overlay of the mask on the image with color blending.

Use case: Visual quality inspection of data prepared for nnU-Net training.
"""

# Paths to the image and mask directories in NIfTI format
images_dir = "nnUnetDataFormat/nnUNet_raw_data_base/nnUNet_raw_data/Task000_STC/imagesTr"
masks_dir  = "nnUnetDataFormat/nnUNet_raw_data_base/nnUNet_raw_data/Task000_STC/labelsTr"

def load_nifti(file_path):
    """Load a NIfTI file and return its data as a NumPy array."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def visualize_sample(image_file, mask_file):
    """
    Display the image and mask side by side, including an overlay.
    
    :param image_file: File name of the input image.
    :param mask_file: File name of the corresponding mask.
    """
    image_path = os.path.join(images_dir, image_file)
    mask_path  = os.path.join(masks_dir, mask_file)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return

    # Load image and mask
    image = load_nifti(image_path)
    mask  = load_nifti(mask_path)

    # Assume the image is 3D and extract the central slice
    central_slice = image.shape[2] // 2
    image_slice = image[:, :, central_slice]
    mask_slice  = mask[:, :, central_slice]

    # Create figure with 3 subplots
    plt.figure(figsize=(12, 6))

    # 1. Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"Original Image:\n{image_file}")
    plt.axis("off")

    # 2. Segmentation mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_slice, cmap="gray")
    plt.title(f"Segmentation Mask:\n{mask_file}")
    plt.axis("off")

    # 3. Overlay of image and mask
    plt.subplot(1, 3, 3)
    plt.imshow(image_slice, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.5)
    plt.title(f"Overlay:\n{image_file} + {mask_file}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example: visualize the first 15 image-mask pairs
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
mask_files  = sorted([f for f in os.listdir(masks_dir)  if f.endswith('.nii.gz')])

for img_file, mask_file in zip(image_files[:15], mask_files[:15]):
    visualize_sample(img_file, mask_file)

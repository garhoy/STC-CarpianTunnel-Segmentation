import os
import numpy as np
from PIL import Image
import glob

"""
This script converts annotated segmentation images (in color) into RGB masks
with predefined colors for each class. Any pixel not matching is assigned to the background (class 0).

Usage:
- Converts all `.png` images in a directory (e.g., test annotations).
- Output masks are saved with RGB colors, useful for visualization or model input.
"""
# --- Class color definitions ---
reference_colors = {
    "background": (0,   0,   0),
    "nerve":      (0, 255, 255),
    "edge":       (255, 0, 255),
    "ligament":   (255, 128, 114),
    "bone":       (255, 255, 0)
}

color2label = {
    "background": 0,
    "nerve":      1,
    "edge":       2,
    "ligament":   3,
    "bone":       4
}

visual_colors = {
    0: [0, 0, 0],       # Background -> Black
    1: [0, 255, 255],   # Nerve -> Cyan
    2: [255, 0, 255],   # Edge -> Fuchsia
    3: [255, 128, 114], # Ligament -> Pink
    4: [255, 255, 0]    # Bone -> Yellow
}

# Prepare arrays for color comparison
color_names = ["background", "nerve", "edge", "ligament", "bone"]
ref_array = np.array([reference_colors[name] for name in color_names], dtype=np.float32)
label_array = np.array([color2label[name] for name in color_names], dtype=np.uint8)

def convert_image_vectorized(input_path, output_path, threshold=20):
    """
    Converts a color-annotated image to a label mask using Euclidean distance.
    Pixels outside the threshold are assigned to background.
    Output is saved as an RGB visualization using predefined colors.
    """
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)

    expanded_img = img_np[:, :, None, :].astype(np.float32)      # (H, W, 1, 3)
    expanded_ref = ref_array[None, None, :, :]                   # (1, 1, C, 3)

    dist = np.sqrt(np.sum((expanded_img - expanded_ref) ** 2, axis=-1))  # (H, W, C)
    min_dist = np.min(dist, axis=-1)     # Closest distance per pixel
    min_idx  = np.argmin(dist, axis=-1)  # Closest class per pixel

    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    valid = (min_dist <= threshold)
    mask[valid] = label_array[min_idx[valid]]

    # Convert label mask to RGB visualization
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in visual_colors.items():
        mask_rgb[mask == label] = color

    out_img = Image.fromarray(mask_rgb)
    out_img.save(output_path)

def process_folder(input_dir, output_dir, threshold=20):
    """
    Converts all PNG images in the input directory and saves processed RGB masks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        convert_image_vectorized(img_path, output_path, threshold)
        print(f"Processed: {img_path} -> {output_path}")

if __name__ == "__main__":
    val_input_dir = "Data/preprocessed_test_masks2/"
    val_output_dir = "Data/mask_test/"

    threshold = 20  # Adjust if needed

    process_folder(val_input_dir, val_output_dir, threshold)

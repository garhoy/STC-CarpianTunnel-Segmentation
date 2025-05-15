import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt


"""
This script compares original annotation masks with preprocessed masks (option 2) 
by displaying them side by side for visual inspection. It also groups images by resolution 
to help identify how many samples share the same shape, useful for preprocessing analysis.
"""


# Paths to CSV and image folders
CSV_PATH = "Data/Test_data.csv"
BASE_ANNOTATION_PATH = "Data/test"
PREPROC_MASKS_OPTION2 = "Data/preprocessed_test_images2"

# Load CSV
df = pd.read_csv(CSV_PATH)

def visualize_masks(index):
    """
    Display two side-by-side masks:
      - Left: preprocessed (option 2)
      - Right: original annotation
    Titles show resolution of each image.

    :param index: row index in the dataframe
    """
    row = df.iloc[index]
    annotation_name = row["mask"].strip()

    # Build full paths
    orig_path = os.path.join(BASE_ANNOTATION_PATH, annotation_name)
    opt2_path = os.path.join(PREPROC_MASKS_OPTION2, annotation_name)

    # Check if both images exist
    for path, label in zip([orig_path, opt2_path], ["Original", "Option 2"]):
        if not os.path.exists(path):
            print(f"⚠️ {label} not found: {path}")
            return

    # Read images (in color to preserve annotation colors)
    orig_cv = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    opt2_cv = cv2.imread(opt2_path, cv2.IMREAD_COLOR)

    if orig_cv is None:
        print(f"⚠️ Could not load original: {orig_path}")
        return
    if opt2_cv is None:
        print(f"⚠️ Could not load option 2: {opt2_path}")
        return

    # Convert BGR (OpenCV) to RGB (matplotlib)
    orig_rgb = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    opt2_rgb = cv2.cvtColor(opt2_cv, cv2.COLOR_BGR2RGB)

    # Get size strings
    size_orig = f"{orig_rgb.shape[1]}x{orig_rgb.shape[0]}"
    size_opt2 = f"{opt2_rgb.shape[1]}x{opt2_rgb.shape[0]}"

    # Show both side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(opt2_rgb)
    axes[0].set_title(f"Option 2\n{annotation_name}\nSize: {size_opt2}")
    axes[0].axis("off")

    axes[1].imshow(orig_rgb)
    axes[1].set_title(f"Original\n{annotation_name}\nSize: {size_orig}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def group_samples_by_resolution():
    """
    Groups images by their resolution (width × height).
    Returns a dict: {(w,h): [indices in df]}.
    """
    groups = {}
    for idx, row in df.iterrows():
        annotation_name = row["mask"].strip()
        annotation_path = os.path.join(BASE_ANNOTATION_PATH, annotation_name)
        if not os.path.exists(annotation_path):
            print(f"⚠️ Annotation not found: {annotation_path}")
            continue
        img_cv = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            print(f"⚠️ Could not load image: {annotation_path}")
            continue
        h, w = img_cv.shape[:2]
        res = (w, h)
        groups.setdefault(res, []).append(idx)
    return groups

def main():
    groups = group_samples_by_resolution()

    print("Found resolutions and sample counts:")
    for res, indices in groups.items():
        print(f"Resolution {res[0]}x{res[1]}: {len(indices)} samples")

    # Show 5 examples per resolution (adjust as needed)
    for res, indices in groups.items():
        print(f"\nShowing samples for resolution {res[0]}x{res[1]}:")
        for idx in indices[:5]:
            visualize_masks(idx)

if __name__ == "__main__":
    main()

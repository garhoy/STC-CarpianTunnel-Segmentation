import os
import cv2
import numpy as np
import pandas as pd

"""
This script preprocesses images and their corresponding segmentation masks 
based on two options:
  1. Crop to centered square and resize.
  2. Stretch vertical images to a fixed horizontal ratio, then resize.

It reads image-mask pairs from a CSV and saves the processed outputs 
to specified folders.
"""


class Preprocess:
    def __init__(self, option=1, desired_size=(512, 512), horizontal_ratio=None):
        self.option = option
        self.desired_size = desired_size
        self.horizontal_ratio = horizontal_ratio if horizontal_ratio else 1.35

    def crop_to_square(self, image, mask):
        """
        Option 1: Crop central square from image and mask, then resize.
        """
        h, w = image.shape[:2]
        side = min(h, w)
        start_x = (w - side) // 2
        start_y = (h - side) // 2
        img_crop = image[start_y:start_y+side, start_x:start_x+side]
        mask_crop = mask[start_y:start_y+side, start_x:start_x+side]
        img_crop = cv2.resize(img_crop, self.desired_size, interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(mask_crop, self.desired_size, interpolation=cv2.INTER_NEAREST)
        return img_crop, mask_crop

    def resize_vertical_to_horizontal_ratio(self, image, mask):
        """
        Option 2: If image is vertical, stretch width to match target aspect ratio, then resize.
        """
        h, w = image.shape[:2]
        ratio = w / h
        if ratio < self.horizontal_ratio:
            new_w = int(self.horizontal_ratio * h)
            image = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_AREA)
            mask  = cv2.resize(mask,  (new_w, h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, self.desired_size, interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask,  self.desired_size, interpolation=cv2.INTER_NEAREST)
        return image, mask

    def preprocess_images_from_csv(self, csv_file, images_folder, masks_folder, out_img_folder, out_mask_folder):
        """
        Load pairs from CSV and apply selected preprocessing option. Save to output folders.
        """
        os.makedirs(out_img_folder, exist_ok=True)
        os.makedirs(out_mask_folder, exist_ok=True)
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_name = row["image"]
            mask_name = row["mask"]
            img_path = os.path.join(images_folder, img_name)
            mask_path = os.path.join(masks_folder, mask_name)

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Skipping {img_name}: missing image or mask.")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

            if self.option == 1:
                processed_img, processed_mask = self.crop_to_square(img, mask)
            elif self.option == 2:
                processed_img, processed_mask = self.resize_vertical_to_horizontal_ratio(img, mask)
            else:
                print(f"Invalid option {self.option}. Skipping.")
                continue

            cv2.imwrite(os.path.join(out_img_folder, img_name), processed_img)
            cv2.imwrite(os.path.join(out_mask_folder, mask_name), processed_mask)
            print(f"Processed: {img_name}")

def main():
    option = 2
    size = (512, 512)
    # Average horizontal ratio based on sample sizes
    horizontal_ratio = (1024/688 + 864/704 + 984/696 + 928/744) / 4
    print("Computed horizontal ratio:", horizontal_ratio)

    csv_file = "Data/Test_data.csv"
    img_dir  = "Data/test"
    msk_dir  = "Data/mask_test"
    out_imgs = "Data/preprocessed_test_images" + str(option)
    out_msks = "Data/preprocessed_test_masks" + str(option)

    pre = Preprocess(option=option, desired_size=size, horizontal_ratio=horizontal_ratio)
    pre.preprocess_images_from_csv(csv_file, img_dir, msk_dir, out_imgs, out_msks)

if __name__ == "__main__":
    main()

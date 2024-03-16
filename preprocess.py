import os
import torch
import cv2
import random
import argparse

import pandas as pd
import numpy as np
from skimage import io, filters, color, morphology
from scipy import ndimage
from PIL import Image
from utils import normalize_percentile_to_255

def img_remove_artifact(img, min_size, area_threshold):
    threshold_binary = img > filters.threshold_multiotsu(img, classes=2)
    threshold_binary = morphology.binary_erosion(threshold_binary)
    keep_mask = morphology.remove_small_objects(threshold_binary, min_size=min_size)
    keep_mask = morphology.remove_small_holes(keep_mask, area_threshold=area_threshold)
    
    img_filter = np.multiply(img, keep_mask)
    
    return img_filter, keep_mask

def canny_edge_detector(image, low_threshold=30, high_threshold=50, kernel_size=3):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply the Canny edge detector
    magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    magnitude = normalize_percentile_to_255(magnitude)

    return magnitude

def main(args):
    image_folder = os.path.join(args.data_directory, args.domain, "images")
    contour_folder = os.path.join(args.data_directory, args.domain, "contours")
    os.makedirs(contour_folder, exist_ok=True)

    img_name_list = []
    contour_name_list = []
    
    for f in os.listdir(image_folder):
        img = Image.open(os.path.join(image_folder, f)).convert("L")
        np_img = np.array(img)

        if args.remove_artifact:
            np_img, keep_mask = img_remove_artifact(np_img, args.min_size, args.area_threshold)

        np_contour = canny_edge_detector(np_img, args.low_threshold, args.high_threshold, args.kernel_size)

        contour = Image.fromarray(np_contour)

        contour.save(os.path.join(contour_folder, f))

        img_name_list.append(f)
        contour_name_list.append(f)

    df_meta = pd.DataFrame({
        "image_name": img_name_list,
        "contour_name": contour_name_list
    })

    df_meta.to_csv(os.path.join(args.data_directory, args.domain, "df_meta.csv"))    

if __name__ == "__main__":
    # Parse args:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)

    ## Remove non-significant artifacts
    parser.add_argument('--remove_artifact', action='store_true')
    parser.add_argument('--min_size', type=int, default=8000)
    parser.add_argument('--area_threshold', type=int, default=3000)

    ## Canny edge
    parser.add_argument('--low_threshold', type=int, default=80)
    parser.add_argument('--high_threshold', type=int, default=150)
    parser.add_argument('--kernel_size', type=int, default=3)
    
    args = parser.parse_args()

    main(args)
    
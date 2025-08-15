import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path

def remove_labels(img_rgb, mode_th=127, kernel_size_opening=10, kernel_size_closing=10, iter_opening=2, iter_closing=2):
    # convert RGB PIL to grayscale numpy
    img_rgb_np = np.array(img_rgb)
    img_grey = cv.cvtColor(img_rgb_np, cv.COLOR_RGB2GRAY)
    
    # apply Otsu thresholding
    _, img_th = cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # if the mode of the greyscale image is closer to 255 (i.e. white background), invert the thresholded image
    hist = cv.calcHist([img_grey], [0], None, [256], [0, 256])
    mode = np.argmax(hist)
    if mode > mode_th:
        img_th = cv.bitwise_not(img_th)
    
    # apply morphological opening to remove labels
    kernel_opening = np.ones((kernel_size_opening, kernel_size_opening), np.uint8)
    opening = cv.morphologyEx(img_th, cv.MORPH_OPEN, kernel_opening, iterations=iter_opening)
    
    # apply morphological closing to fill small holes within white body parts
    kernel_closing = np.ones((kernel_size_closing, kernel_size_closing), np.uint8)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_closing, iterations=iter_closing)
    
    # apply floodfill to fill remaining large holes within white body parts
    filled = closing.copy()
    contours, hierarchy = cv.findContours(filled, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # hierarchy:  3D array [[[next, prev, first child, parent]]], access inner 2 dimensions
        for idx, contour in enumerate(contours):
            if hierarchy[idx][3] != -1: # -1 means no parent (i.e. outer contour)
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = contour[0][0] # use the first point in the contour as centroid if zero moments
                cv.floodFill(filled, None, (cX, cY), 255, loDiff=0, upDiff=0)
    
    # apply the mask to the RGB image
    mask = filled.astype(np.float32)/255.0
    mask_3ch = mask[..., None]
    img_masked = (img_rgb_np.astype(np.float32)*mask_3ch).astype(np.uint8)
    
    return img_masked # img_grey, img_th, opening, closing, filled

def process_folder(processed_root, split="train"):
    processed_root = Path(processed_root)
    png_paths = list(processed_root.joinpath(split).rglob("*.png"))

    print(f"Found {len(png_paths)} images in {processed_root}/{split}")
    for p in png_paths:
        try:
            img = Image.open(p).convert("RGB")
            proc = remove_labels(img, mode_th=127, kernel_size_opening=10, kernel_size_closing=10, iter_opening=2, iter_closing=2)
            # save back as 8‑bit grayscale PNG
            Image.fromarray(proc).save(p)
        except Exception as e:
            print(f"Failed on {p}: {e}")

if __name__ == "__main__":
    process_folder("MURA-v1.1-processed", split="train")
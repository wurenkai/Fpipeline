import cv2
import numpy as np
import os
from glob import glob


mask_L_path = 'left_mask.png'
mask_R_path = 'right_mask.png'
input_folder = ''
output_folder = ''
os.makedirs(output_folder, exist_ok=True)


def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    mask_3ch = cv2.merge([mask, mask, mask]) // 255
    return mask_3ch, center, radius


mask_L, center_L, radius_L = preprocess_mask(mask_L_path)
mask_R, center_R, radius_R = preprocess_mask(mask_R_path)


image_paths = glob(os.path.join(input_folder, '*.*'))

for img_path in image_paths:
    filename = os.path.basename(img_path)
    eye_img = cv2.imread(img_path)
    if eye_img is None:
        print(f"skip: {filename}")
        continue


    if '_L' in filename:
        mask, center, radius = mask_L, center_L, radius_L
    elif '_R' in filename:
        mask, center, radius = mask_R, center_R, radius_R
    else:
        print(f"skip: {filename}")
        continue


    diameter = 2 * radius
    resized_eye = cv2.resize(eye_img, (diameter, diameter))


    background = np.zeros(mask.shape, dtype=np.uint8)


    x1, y1 = center[0] - radius, center[1] - radius
    x2, y2 = center[0] + radius, center[1] + radius


    background[y1:y2, x1:x2] = resized_eye


    fused = cv2.bitwise_and(background, mask * 255)


    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, fused)
    print(f"completed: {filename}")

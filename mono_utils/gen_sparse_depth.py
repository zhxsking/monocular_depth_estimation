# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from boxx import show

# OB
# path_img = r"C:\pic\ob_dataset\sync\baseline_large\outdoor\FactoryDistrict\1RgbLeftLarge.jpg"
# path_depth = r"C:\pic\ob_dataset\sync\baseline_large\outdoor\FactoryDistrict\1DepthLeftLarge.png"

# NYU
path_img = r"D:\pic\NYU-depth\BTS_data\sync\dining_room_0013\rgb_00203.jpg"
path_depth = r"D:\pic\NYU-depth\BTS_data\sync\dining_room_0013\sync_depth_00203.png"

# KITTI
# path_img = r"D:\pic\KITTI\sync\data_depth\test_depth_completion_anonymous\image\0000000999.png"
# path_depth = r"D:\pic\KITTI\sync\data_depth\test_depth_completion_anonymous\velodyne_raw\0000000999.png"

img = np.array(Image.open(path_img))
depth = np.array(Image.open(path_depth), dtype=np.uint16)

# NYU
depth_proj = depth.astype(np.float32) / 1000 * 256
depth_proj = depth_proj.astype(np.uint16)
# OB
# depth[depth > 5000] = 5000
# depth_proj = depth.astype(np.float32) / 100 * 256
# depth_proj = depth_proj.astype(np.uint16)
# KITTI
# depth_proj = depth


crop_h = 352
crop_w = 1216

seq1 = iaa.Sequential([
    iaa.CenterPadToFixedSize(height=crop_h, width=crop_w),  # 保证可crop
    iaa.CenterCropToFixedSize(height=crop_h, width=crop_w),
], random_order=True)

seq2 = iaa.Sequential([
    # iaa.CoarseDropout(0.19, size_px=200),
    iaa.Dropout(1-0.05),
], random_order=True)

img_aug = seq1(image=img)
# depth_aug = seq1(image=depth_proj)
depth_aug = seq1(image=seq2(image=depth_proj))
show(img_aug)
show(depth)
show(depth_aug)

name = '\\nyu-1-kitti.png'
cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\image'+name, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\velodyne_raw'+name, depth_aug)
cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\groundtruth_depth'+name, depth)

cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\image'+name, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\velodyne_raw'+name, d_a)
cv2.imwrite(r'D:\pic\KITTI\persudo\data_depth\test_depth_completion_anonymous\groundtruth_depth'+name, depth_aug)
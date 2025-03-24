from MVSFormerPlusPlus.datasets.data_io import read_pfm, unpack_float32
import os
import cv2
import torch
from matplotlib import cm
from PIL import Image
import numpy as np

scene = "small_wooden_chessboard"
exp_name = "orig_MVSPP" # MVSD++_train_20250314_193649
vid = 4

dmin = 0.473
dmax = 0.983

path_to_depth = f"MVSFormerPlusPlus/saved/models/DINOv2/{exp_name}/test/_2368x1920/{scene}/depth_est/{vid:08d}.pfm"
path_to_gt = '/sk3d/addons/{}/proj_depth/stl.clean_rec.aa@tis_right.undist/{:0>4}.png'.format(scene, vid)

if not os.path.exists(f"MVSFormerPlusPlus/saved/models/DINOv2/{exp_name}/test/_2368x1920/{scene}/depth_est_png/"):
    os.makedirs(f"MVSFormerPlusPlus/saved/models/DINOv2/{exp_name}/test/_2368x1920/{scene}/depth_est_png/")

path_save_depth = f"MVSFormerPlusPlus/saved/models/DINOv2/{exp_name}/test/_2368x1920/{scene}/depth_est_png/"

# load gt depth
depth_gt = unpack_float32(np.asarray(Image.open(path_to_gt))).copy()
# depth_gt = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min())
# print(depth_gt.shape, depth_gt)
# depth_gt = Image.fromarray(depth_gt.astype("uint8"))
# depth_gt.save(os.path.join(path_save_depth, f"{vid:08d}_gt.png"))
depth_gt = (depth_gt - dmin) / (dmax - dmin)
depth_gt = cm.plasma(depth_gt)[...,:3]
depth_gt = np.stack([depth_gt[...,2], depth_gt[...,1], depth_gt[...,0]], axis=2) # BGR
cv2.imwrite(os.path.join(path_save_depth, f"{vid:08d}_gt.png"), depth_gt * 255)

# load predicted depth
data, scale = read_pfm(path_to_depth)
# data = (data - data.min()) / data.max()
# data = Image.fromarray(data.astype("uint8"))
# data.save(os.path.join(path_save_depth, f"{vid:08d}.png"))
data = (data - dmin) / (dmax - dmin)
img = cm.plasma(data)[...,:3]
img = np.stack([img[...,2], img[...,1], img[...,0]], axis=2) # BGR
cv2.imwrite(os.path.join(path_save_depth, f"{vid:08d}.png"), img * 255)
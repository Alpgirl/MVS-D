from MVSFormerPlusPlus.datasets.data_io import read_pfm, unpack_float32
import os
import cv2
import torch
from matplotlib import cm
from PIL import Image
import numpy as np

scene = "white_box"
exp_name = "orig_MVSPP" # MVSD++_train_20250314_193649
vid = 4

dmin = 0.473
dmax = 0.983

path_to_depth = f"MVSFormerPlusPlus/saved/models/DINOv2/{exp_name}/test/_2368x1920/{scene}/depth_est/{vid:08d}.pfm"
path_to_gt = '/sk3d/addons/{}/proj_depth/stl.clean_rec.aa@tis_right.undist/{:0>4}.png'.format(scene, vid)

# load gt depth
depth_gt = unpack_float32(np.asarray(Image.open(path_to_gt))).copy()
depth_gt = (depth_gt - dmin) / (dmax - dmin)
depth_gt = cm.plasma(depth_gt)[...,:3]
depth_gt = np.stack([depth_gt[...,2], depth_gt[...,1], depth_gt[...,0]], axis=2) # BGR
cv2.imwrite("depth_map_gt.png", depth_gt*255)

# load predicted depth
data, scale = read_pfm(path_to_depth)
data = (data - dmin) / (dmax - dmin)
img = cm.plasma(data)[...,:3]
img = np.stack([img[...,2], img[...,1], img[...,0]], axis=2) # BGR
cv2.imwrite("depth_map.png", img*255)
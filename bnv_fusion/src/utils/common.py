import cv2
import imageio
import numpy as np
import os
import os.path as osp
import torch
import time
import skimage
from skimage import transform
from PIL import Image


class Timer:
    def __init__(self, names):
        self.times = {n: 0 for n in names}
        self.t0 = {n: 0 for n in names}

    def start(self, name):
        self.t0[name] = time.time()
    
    def log(self, name):
        self.times[name] += time.time() - self.t0[name]


def to_cuda(in_dict):
    for k in in_dict:
        if isinstance(in_dict[k], torch.Tensor):
            in_dict[k] = in_dict[k].to("cuda")


def to_cpu(in_dict):
    for k in in_dict:
        if isinstance(in_dict[k], torch.Tensor):
            in_dict[k] = in_dict[k].cpu()


def override_weights(model, pretrained_weights, keys):
    """
    Args:
        model: pytorch nn module
        pretrained_weights: OrderedDict of state_dict
        keys: a list of keyword. the weights to be overrided if matched 
    """

    pretrained_dict = {}
    for model_key in model.state_dict().keys():
        if any([(key in model_key) for key in keys]):
            if model_key not in pretrained_weights:
                print(f"[warning]: {model_key} not in pretrained weight")
                continue
            pretrained_dict[model_key] = pretrained_weights[model_key] 
    model.load_state_dict(pretrained_dict, strict=False)


def get_file_paths(dir, file_type=None):
    names = sorted(os.listdir(dir))
    out = []
    for n in names:
        if os.path.isdir(osp.join(dir, n)):
            paths = get_file_paths(osp.join(dir, n), file_type)
            out.extend(paths)
        else:
            if file_type is not None:
                if n.endswith(file_type):
                    out.append(osp.join(dir, n))
            else:
                out.append(osp.join(dir, n))
    return out


def inverse_sigmoid(x):
    return np.log(x) - np.log(1-x)


def load_rgb(path, downsample_scale=0):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downsample_scale > 0:
        img = transform.rescale(img, (downsample_scale, downsample_scale, 1))
    # pixel values between [-1,1]
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_depth(
    path,
    downsample_scale,
    downsample_mode="dense",
    max_depth=None,
    add_noise=False
):
    depth = cv2.imread(path, -1) / 1000.
    if downsample_scale > 0:
        img_h, img_w = depth.shape
        if downsample_mode == "dense":
            reduced_w = int(img_w * downsample_scale)
            reduced_h = int(img_h * downsample_scale)
            depth = cv2.resize(
                depth,
                dsize=(reduced_w, reduced_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            assert downsample_mode == "sparse"
            downsample_mask = np.zeros_like(depth)
            interval = int(1 / downsample_scale)
            downsample_mask[::interval, ::interval] = 1
            depth = depth * downsample_mask
    mask = depth > 0
    if max_depth is not None:
        mask *= depth < max_depth
        depth = depth * mask
    if add_noise:
        noise_depth = noise_simulator.simulate(depth)
        # noise_depth = add_depth_noise(depth)
        noise_depth = noise_depth * mask
        return depth, noise_depth, mask
    else:
        return depth, depth, mask
    
    
def unpack_float32(ar):
    r"""Unpacks an array of uint8 quadruplets back to the array of float32 values.
    Parameters
    ----------
    ar : np.naddary
        of shape [**, 4].
    Returns
    -------
    ar : np.naddary
        of shape [**]
    """
    shape = ar.shape[:-1]
    return ar.ravel().view(np.float32).reshape(shape)
    

def load_depth_sk3d(
    path,
    downsample_scale,
    downsample_mode="dense",
    max_depth=None,
    add_noise=False
):
    depth = unpack_float32(np.asarray(Image.open(path))).copy()
    mask_ = (depth >= 0).astype(np.float32)
    depth = np.where(mask_ != 0, depth, 0)
    if downsample_scale > 0:
        img_h, img_w = depth.shape
        if downsample_mode == "dense":
            reduced_w = int(img_w * downsample_scale)
            reduced_h = int(img_h * downsample_scale)
            depth = cv2.resize(
                depth,
                dsize=(reduced_w, reduced_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            assert downsample_mode == "sparse"
            downsample_mask = np.zeros_like(depth)
            interval = int(1 / downsample_scale)
            downsample_mask[::interval, ::interval] = 1
            depth = depth * downsample_mask
    mask = depth > 0
    if max_depth is not None:
        mask *= depth < max_depth
        depth = depth * mask
    if add_noise:
        noise_depth = noise_simulator.simulate(depth)
        # noise_depth = add_depth_noise(depth)
        noise_depth = noise_depth * mask
        return depth, noise_depth, mask
    else:
        return depth, depth, mask

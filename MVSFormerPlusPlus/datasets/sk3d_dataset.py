import os
import cv2
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.data_io import *

# bnv-fusion
from kornia.geometry.depth import depth_to_normals, depth_to_3d_v2

import bnv_fusion.src.utils.geometry as geometry
import copy
import warnings

from .color_jittor import ColorJitter

s_h, s_w = 0, 0
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Since kornia 0.8.0 the `depth_to_3d` is deprecated.*")

class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img, gamma):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


class Sk3DDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, num_depths, interval_scale, random_crop=False, augment=False, aug_args=None, **kwargs):
        super(Sk3DDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.num_depths = num_depths
        self.interval_scale = interval_scale
        self.crop, self.resize = kwargs.get('crop', None), kwargs.get('resize', None) # in casmvs i used "resize": [576, 768], now it's multiscale

        if mode != 'train':
            self.random_crop = False
            self.augment = False
        else:
            self.random_crop = random_crop
            self.augment = augment

        self.fix_res = kwargs.get("fix_res", False)  # whether to fix the resolution of input image.
        self.fix_wh = False
        self.kwargs = kwargs

        if mode != 'train':
            self.random_crop = False
        else:
            self.random_crop = random_crop

        self.rgbd = kwargs.get('rgbd', False)
        self.sensor_depth_resize = kwargs.get('sensor_depth_resize', self.resize) # this is resize for NeuralFusion expected input size

        self.multi_scale = kwargs.get('multi_scale', False)
        self.multi_scale_args = kwargs['multi_scale_args']
        
        self.scales = self.multi_scale_args['scales']#[::-1]
        self.resize_range = self.multi_scale_args['resize_range']

        self.batch_size = kwargs.get('batch_size', 4) if mode == 'train' else kwargs.get('eval_batch_size', 4)
        self.world_size = kwargs.get('world_size', 1)

        if self.augment and mode == 'train':
            self.color_jittor = ColorJitter(brightness=aug_args['brightness'], contrast=aug_args['contrast'],
                                            saturation=aug_args['saturation'], hue=aug_args['hue'])
            self.to_tensor = transforms.ToTensor()
            self.random_gamma = RandomGamma(min_gamma=aug_args['min_gamma'], max_gamma=aug_args['max_gamma'], clip_image=True)
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if mode == 'train' or mode == 'val':
            self.light_types = [
                            'flash@best']#, 'flash@fast', 'ambient@best', 'ambient_low@fast', 'hard_left_bottom_close@best',
                            # 'hard_left_bottom_far@best', 'hard_left_top_close@best', 'hard_left_top_far@best', 'hard_right_bottom_close@best',
                            # 'hard_right_top_close@best', 'hard_right_top_far@best', 'soft_left@best', 'soft_right@best', 'soft_top@best']
        else:
            self.light_types = ['ambient@best']
        self.metas = self.build_list()
        self.list_begin = []
        self.dimensions = self.calculate_dimensions()


    def calculate_dimensions(self):
        bounds = np.load("/app/bounds_v1.npy")
        return bounds
    

    def build_list(self):
        metas = []  # {}
        if type(self.listfile) is list:
            scans = self.listfile
        else:
            with open(self.listfile, 'r') as f:
                scans = f.readlines()
                scans = [s.strip() for s in scans]

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "addons/{}/tis_right/rgb/mvsnet_input/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(1):#num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        src_views = src_views[:(self.nviews - 1)]

                        for light_type in self.light_types:
                            metas.append((scan, ref_view, light_type, src_views, scan))
                    

        self.interval_scale = interval_scale_dict
        print(metas)
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas
        
    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        depth_interval *= interval_scale

        if self.mode == 'train' or self.mode == 'val':
            crop_h_start, crop_w_start = 601, 291
        else:
            crop_h_start, crop_w_start = 576, 289

        intrinsics[0, 2] -= crop_w_start
        intrinsics[1, 2] -= crop_h_start

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename).convert('RGB')  # 0~255
        img = np.asarray(img)

        # crop in the way that all scenes are inside these bounds. To accelerate train/val/test
        if self.mode == 'train' or self.mode == 'val':
            crop_h_start, crop_h_end, crop_w_start, crop_w_end = [601, 1952, 291, 1887]
        else:
            crop_h_start, crop_h_end, crop_w_start, crop_w_end = [576, 1952, 289, 1889]
        
        img = img[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return img

    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)
            else:
                depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), 
                                                        (int(ori_h * resize_scale), int(ori_w * resize_scale)),
                                                        mode='nearest', recompute_scale_factor=False)[0][0]

        if mask is not None:
            if isinstance(depth, np.ndarray):
                mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)
            else:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                        (int(ori_h * resize_scale), int(ori_w * resize_scale)),
                                                        mode='nearest', recompute_scale_factor=False)[0][0]


        return img, depth, output_intrinsics, mask

    def final_crop(self, img, depth, intrinsic, mask, crop_h, crop_w, offset_y=None, offset_x=None):
        h, w, _ = img.shape
        if offset_x is None or offset_y is None:
            if self.random_crop:
                offset_y = random.randint(0, h - crop_h)
                offset_x = random.randint(0, w - crop_w)
            else:
                offset_y = (h - crop_h) // 2
                offset_x = (w - crop_w) // 2
        cropped_image = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w, :]

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        if depth is not None:
            cropped_depth = depth[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_depth = None

        if mask is not None:
            cropped_mask = mask[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_mask = None

        return cropped_image, cropped_depth, output_intrinsics, cropped_mask
    
    def generate_stages_array(self, arr):
        h, w = arr.shape
        arr_ms = {
            "stage1": torch.nn.functional.interpolate(arr.view(1, 1, h, w), (h // 8, w // 8), mode='nearest', recompute_scale_factor=False)[0][0],
            "stage2": torch.nn.functional.interpolate(arr.view(1, 1, h, w), (h // 4, w // 4), mode='nearest', recompute_scale_factor=False)[0][0],
            "stage3": torch.nn.functional.interpolate(arr.view(1, 1, h, w), (h // 2, w // 2), mode='nearest', recompute_scale_factor=False)[0][0],
            "stage4": arr
        }
        return arr_ms

    def read_depth_and_mask(self, filename):
        depth = unpack_float32(np.asarray(Image.open(filename))).copy()
        depth = torch.FloatTensor(depth)

        ## CROP that fits every scene
        if self.mode == 'train' or self.mode == 'val':
            crop_h_start, crop_h_end, crop_w_start, crop_w_end = [601, 1952, 291, 1887]
        else:
            crop_h_start, crop_h_end, crop_w_start, crop_w_end = [576, 1952, 289, 1889]
        
        depth = depth[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        mask = (depth >= 0).to(torch.float32)

        return depth, mask

    def scale_depth_input(self, depth, mask, crop, resize, intrinsics=None):
        if crop is not None:
            crop_h_start, crop_h_end, crop_w_start, crop_w_end = crop
            depth = depth[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

            if intrinsics is not None:
                intrinsics[0, 2] -= crop_x_start  
                intrinsics[1, 2] -= crop_y_start  

        h, w = depth.shape[:2]

        if resize is not None:
            new_h, new_w = resize
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='nearest',
                                                recompute_scale_factor=False)[0][0]

            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='nearest',
                                                recompute_scale_factor=False)[0][0]
            
            scale_w = 1.0 * new_w / w
            scale_h = 1.0 * new_h / h
            
            if intrinsics is not None:
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h
            
        if intrinsics is not None:
            return depth, mask, intrinsics
        else:
            return depth, mask
        

    def bnv_sensor_depth_item(self, depth, rgb, mask, T_cw, intr_mat):
        """
        input:
            depth: torch.Tensor, sensor depth
            rgb: torch.Tensor, rgb image
            mask: torch.Tensor, sensor mask
            T_cw: numpy.ndarray, extrinsics 4x4
            intr_mat: numpy.ndarray, intrinsics 3x3

        return:
            frame: dict, 
                        rgbd: numpy.ndarray, rgb+sensor depth
                        gt_pts, numpy.ndarray, sensor depth points in world xyz
                        input_pts, numpy.ndarray, gt_pts + normals where depth > 0
        """
        mask = mask.to(torch.bool)
        normal = depth_to_normals(
            torch.unsqueeze(torch.unsqueeze(depth, 0), 0),
            torch.unsqueeze(torch.from_numpy(intr_mat), 0)
        )[0].permute(1, 2, 0).numpy()

        gt_xyz_map = depth_to_3d_v2(
            depth,
            torch.from_numpy(intr_mat)
        ).numpy()

        img_h, img_w = depth.shape
        gt_xyz_map_w = (T_cw @ geometry.get_homogeneous(gt_xyz_map.reshape(-1, 3)).T)[:3, :].T
        gt_xyz_map_w = gt_xyz_map_w.reshape(img_h, img_w, 3)

        # NOTE: VERY IMPORTANT TO * -1 for normal due to a bug in data preparation in
        # data_prepare_depth_shapenet.py!

        normal_w = (T_cw[:3, :3] @ normal.reshape(-1, 3).T).T
        rgbd = np.concatenate([rgb, depth[None, ...]], axis=0)

        pts_c = geometry.depth2xyz(depth.numpy(), intr_mat).reshape(-1, 3)
        pts_w_frame = (T_cw @ geometry.get_homogeneous(pts_c).T)[:3, :].T
        input_pts = np.concatenate(
            [pts_w_frame, normal_w],
            axis=-1
        ) # [1376*1600, 6]
        input_pts = input_pts[mask.reshape(-1)] # [2124147, 6]
        frame = {
            # "depth_path": depth_path,
            # "img_path": image_path,
            # "scene_id": self.scan_id,
            # "frame_id": idx,
            "T_wc": T_cw,
            "intr_mat": intr_mat,
            "rgbd": rgbd,
            # "mask": mask,
            "gt_pts": pts_w_frame,
            # "gt_depth": depth,
            "input_pts": input_pts,
        }
        return frame
    
    
    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1

        # random img size:256~512
        if self.mode == 'train':
            barrel_num = int(len(self.metas) / (self.batch_size * self.world_size))
            barrel_num += 2
            self.img_size_map = np.arange(0, len(self.scales))


    def __getitem__(self, idx):
        global s_h, s_w
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, light_type, src_views, scene_name = meta
        if self.mode == 'train':
            np.random.shuffle(src_views)

        if self.augment:
            fn_idx = torch.randperm(4)
            brightness_factor = torch.tensor(1.0).uniform_(self.color_jittor.brightness[0], self.color_jittor.brightness[1]).item()
            contrast_factor = torch.tensor(1.0).uniform_(self.color_jittor.contrast[0], self.color_jittor.contrast[1]).item()
            saturation_factor = torch.tensor(1.0).uniform_(self.color_jittor.saturation[0], self.color_jittor.saturation[1]).item()
            hue_factor = torch.tensor(1.0).uniform_(self.color_jittor.hue[0], self.color_jittor.hue[1]).item()
            gamma_factor = self.random_gamma.get_params(self.random_gamma._min_gamma, self.random_gamma._max_gamma)

        # scan = scene_name = key
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        depth_ms = None
        mask = None
        proj_matrices = []
        sensor_depths_arr, sensor_masks_arr = [], []
        sensor_intrs, sensor_extrs = [], []
        sensor_rgbds, sensor_gt_pts, sensor_input_pts = [], [], []
        for i, vid in enumerate(view_ids):
            ## LOAD RGB
            img_filename = os.path.join(self.datapath, 'dataset/{}/tis_right/rgb/undistorted/{}/{:0>4}.png'.format(scan, light_type, vid))
            
            img = self.read_img(img_filename) 
            img = np.array(img) # already cropped to remove background
            img_init = copy.deepcopy(img)

            ## LOAD CAMERA PARAMS
            proj_mat_filename = os.path.join(self.datapath, 'addons/{}/tis_right/rgb/mvsnet_input/{:0>8}_cam.txt'.format(scan, vid))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scan]) # already cropped to remove background 
            intrinsics_orig = intrinsics.copy()

            ## LOAD GT DEPTH and MASK
            if i == 0:
                depth_filename = os.path.join(self.datapath, 'addons/{}/proj_depth/stl.clean_rec.aa@tis_right.undist/{:0>4}.png'.format(scan, vid))
                depth, mask = self.read_depth_and_mask(depth_filename) # already cropped to remove background
                depth = depth.where(mask != 0, depth.new_tensor(0))
            else: 
                depth, mask = None, None

            # CALCULATE resize_scale according to crop size
            if self.mode == 'train':
                # multi-scale training with dynamic image sizes and batch size
                [crop_h, crop_w] = self.scales[self.idx_map[idx] % len(self.scales)]
                enlarge_scale = self.resize_range[0] + random.random() * (self.resize_range[1] - self.resize_range[0])
                resize_scale_h = np.clip((crop_h * enlarge_scale) / 1351, 0.45, 1.0)
                resize_scale_w = np.clip((crop_w * enlarge_scale) / 1596, 0.45, 1.0)
                resize_scale = max(resize_scale_h, resize_scale_w)
            elif self.mode == 'val':
                # this is different from dtu_dataset_ms which crops non-resized image
                # permanent size of validation images
                [crop_h, crop_w] = [576, 768]
                enlarge_scale = 1.0
                resize_scale_h = np.clip((crop_h * enlarge_scale) / 1351, 0.45, 1.0)
                resize_scale_w = np.clip((crop_w * enlarge_scale) / 1596, 0.45, 1.0)
                resize_scale = max(resize_scale_h, resize_scale_w)
            else:
                [crop_h, crop_w], resize_scale = [None, None], 1.

            # RESIZE image, depth, mask and alter intrinsics
            if self.mode != "test":
                if resize_scale != 1.0:
                    img, depth, intrinsics, mask = self.pre_resize(img, depth, intrinsics, mask, resize_scale)

                img, depth, intrinsics, mask = self.final_crop(img, depth, intrinsics, mask, crop_h=crop_h, crop_w=crop_w)

            # scale input
            # img, img_intr = self.scale_mvs_input(img, intrinsics, self.crop, self.resize)

            # AUGMENT RGB images
            img = Image.fromarray(img)
            if not self.augment:
                imgs.append(self.transforms(img))
            else:
                img_aug = self.color_jittor(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                img_aug = self.to_tensor(img_aug)
                img_aug = self.random_gamma(img_aug, gamma_factor)
                img_aug = self.normalize(img_aug)
                imgs.append(img_aug)
            
            # COMBINE extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            # GENERATE stage depths and masks for reference view
            if i == 0:  # reference view
                # depth, mask = self.scale_depth_input(depth, mask, self.crop, self.resize)
                depth_ms = self.generate_stages_array(depth)
                mask_ms = self.generate_stages_array(mask)

                depth_max = depth_interval * (self.num_depths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_interval * (self.num_depths - 0.5) + depth_min, depth_interval, dtype=np.float32)

            # READ SENSOR depth
            if self.rgbd:
                sensor_depth_filename = os.path.join(self.datapath, 'addons/{}/proj_depth/kinect_v2.undist@tis_right.undist/{:0>4}.png'.format(scan, vid)) #os.path.join(self.datapath, 'addons/{}/proj_depth/kinect_v2.undist@tis_right.undist/{:0>4}.png'.format(scan, vid))
                # read sensor depth from file
                # depth is reshaped to pretrained NeuralFusion expected input shape
                sensor_depth, sensor_mask = self.read_depth_and_mask(sensor_depth_filename) # already cropped to remove background
                sensor_depth, sensor_mask, sensor_intr = self.scale_depth_input(sensor_depth, sensor_mask, self.crop, self.sensor_depth_resize, intrinsics=intrinsics_orig)
                sensor_depth = sensor_depth.where(sensor_mask != 0, sensor_depth.new_tensor(0))   
                sensor_extr = np.linalg.inv(extrinsics)   

                # print(sensor_mask.dtype)
                img_init = self.transforms(img_init)
                frame = self.bnv_sensor_depth_item(sensor_depth, img_init, sensor_mask, sensor_extr, sensor_intr)

                sensor_depths_arr.append(sensor_depth)
                sensor_masks_arr.append(sensor_mask)
                # sensor_intrs.append(torch.FloatTensor(sensor_intr))
                # sensor_extrs.append(torch.FloatTensor(sensor_extr))
                # sensor_rgbds.append(torch.FloatTensor(frame["rgbd"]))
                sensor_gt_pts.append(torch.FloatTensor(frame["gt_pts"]))
                sensor_input_pts.append(torch.FloatTensor(frame["input_pts"]))
            else:
                sensor_depth, sensor_mask = None, None
            

        # all
        imgs = torch.stack(imgs)  # [V,3,H,W]
        proj_matrices = np.stack(proj_matrices)

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 1.

        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": stage1_pjmats,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats
        }
   
        if self.rgbd:
            sensor_depths_arr = torch.stack(sensor_depths_arr) # [V, H, W]
            sensor_masks_arr = torch.stack(sensor_masks_arr) # [V, H, W]
            # sensor_intrs = torch.stack(sensor_intrs)
            # sensor_extrs = torch.stack(sensor_extrs)
            # sensor_rgbds = torch.stack(sensor_rgbds)

        result = {"imgs": imgs,
                "depth": depth_ms,
                "mask": mask_ms,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "scan": scan,
                "depth_interval": depth_interval, 
                'depths_max': depth_max, 'depths_min': depth_min}
        
        if self.rgbd:
            result["sensor_depths"] = sensor_depths_arr
            result["sensor_depth_masks"] = sensor_masks_arr
            # result["sensor_intr"] = sensor_intrs
            # result["sensor_extr"] = sensor_extrs
            result["input_pts"] = sensor_input_pts
            # result["rgbd"] = sensor_rgbds
            result["gt_pts"] = sensor_gt_pts
        
        return result


        

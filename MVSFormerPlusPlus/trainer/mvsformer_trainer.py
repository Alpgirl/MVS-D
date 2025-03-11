import collections
import os
import pdb
import time
import torch
import cv2
import torch.distributed as dist
from tqdm import tqdm
import torch.utils.checkpoint as cp
from base import BaseTrainer
from models.losses import *
from utils import *

from pytorch_lightning import seed_everything
from bnv_fusion.src.models.fusion.local_point_fusion import LitFusionPointNet
from bnv_fusion.src.run_e2e import NeuralMap

from torch import profiler


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, optimizer, config, bnvconfig, data_loader, valid_data_loader=None, lr_scheduler=None, writer=None,
                 rank=0, ddp=False,
                 train_sampler=None, debug=False):
        super().__init__(model, optimizer, config, writer=writer, rank=rank, ddp=ddp)
        self.config = config
        self.bnvconfig = bnvconfig
        self.ddp = ddp
        self.debug = debug
        self.data_loader = data_loader
        self.train_sampler = train_sampler
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = config['trainer']['logging_every']
        self.depth_type = self.config['arch']['args']['depth_type']
        self.ndepths = self.config['arch']['args']['ndepths']
        self.multi_scale = self.config['data_loader'][0]['args']['multi_scale']
        self.scale_batch_map = self.config['data_loader'][0]['args']['multi_scale_args']['scale_batch_map']
        # self.multi_ratio_config = self.config['data_loader'][0]['args']['multi_scale_args'].get('multi_ratio', None)
        # if self.multi_ratio_config is not None:
        #     self.multi_ratio_scale_batch_map = self.multi_ratio_config['scale_batch_map']
        # else:
        #     self.multi_ratio_scale_batch_map = None

        self.inverse_depth = config['arch']['args']['inverse_depth']

        self.loss_arg = config['arch']['loss']
        self.grad_norm = config['trainer'].get('grad_norm', None)
        self.dataset_name = config['arch']['dataset_name']
        self.train_metrics = DictAverageMeter()
        self.valid_metrics = DictAverageMeter()
        self.loss_downscale = config['optimizer']['args'].get('loss_downscale', 1.0)
        self.scale_dir = True
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False
        self.bf16 = config["arch"].get("bf16", False)

        if 'rgbd' in config['data_loader'][0]['args'].keys():
            self.rgbd = config['data_loader'][0]['args']['rgbd']
        else:
            self.rgbd = False
        
        self.dimensions = data_loader[0].dataset.dimensions


    # @staticmethod
    # @hydra.main(config_path="/app/bnv_fusion/configs/", config_name="config.yaml")
    # def prepare_bnvfusion_config(config: DictConfig):
    #     with open_dict(config):
    #     # Modify existing parameters
    #         config.model = "fusion_pointnet_model"
    #         # dataset=sk3d_dataset dataset.scan_id="sk3d/green_funnels" trainer.checkpoint=$PWD/pretrained/pointnet_tcnn.ckpt model.tcnn_config=$PWD/src/models/tcnn_config.json model.mode="demo"

    #     if "seed" in config.trainer:
    #         seed_everything(config.trainer.seed)

    #     hydra_utils.extras(config)
    #     hydra_utils.print_config(config, resolve=True)

    #     return config


    def prepare_bnvfusion_input(self, sample_cuda, b_start=None, b_end=None):
        sensor_depths = None
        sensor_depths_masks = None
        if b_start is None:
            b_start = 0
        if b_end is None:
            b_end = sample_cuda['sensor_depths'].shape[0]
        
        # print(f"input_pts full: {len(sample_cuda['input_pts'])}, b_start: {b_start}, b_end: {b_end}")
        # prepare sensor data
        if 'sensor_depths' in sample_cuda.keys():
            sensor_data = {
                'depths': sample_cuda['sensor_depths'][b_start:b_end],
                'masks': sample_cuda['sensor_depth_masks'][b_start:b_end],
                # 'intr': sample_cuda['sensor_intr'][b_start:b_end],
                # 'extr': sample_cuda['sensor_extr'][b_start:b_end],
                'input_pts': sample_cuda['input_pts'][b_start:b_end],
                'gt_pts': sample_cuda['gt_pts'][b_start:b_end],
                # 'rgbd': sample_cuda['rgbd'][b_start:b_end]
            }
        else:
            sensor_data = {}

        # prepare config
        if self.bnvconfig.trainer.__check_attr__("seed"):
            seed_everything(self.bnvconfig.trainer.seed)

        # initialize model
        print("initializing model")
        pointnet_model = LitFusionPointNet(self.bnvconfig)
        pretrained_weights = torch.load(self.bnvconfig.trainer.checkpoint)
        pointnet_model.load_state_dict(pretrained_weights['state_dict'])
        pointnet_model.eval()
        pointnet_model.cuda()
        pointnet_model.freeze()

        # # initialize volume object
        print("initializing volume object")
        neural_map = NeuralMap(
            self.dimensions,
            self.bnvconfig,
            pointnet_model,
            working_dir="")

        return sensor_data, pointnet_model, neural_map
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        # if self.ddp:
        # self.train_sampler.set_epoch(epoch)  # Shuffle each epoch
        if self.multi_scale:
            for i in range(len(self.data_loader)):
                self.data_loader[i].dataset.reset_dataset(self.train_sampler)

        self.model.train()
        dist_group = torch.distributed.group.WORLD

        global_step = 0
        scaled_grads = collections.defaultdict(list)
        pre_scale = int(self.scaler.get_scale()) if self.fp16 else None

        # training
        for dl in self.data_loader:
            if self.rank == 0:
                t_loader = tqdm(dl, desc=f"Epoch: {epoch}/{self.epochs}. Train.",
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(dl))
            else:
                t_loader = dl
            for batch_idx, sample in enumerate(t_loader):
                # print("INIT GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                # print("INIT GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
                num_stage = 3 if self.config['data_loader'][0]['args'].get('stage3', False) else 4
                
                ## DEBUG
                # for k, v in sample.items():
                #     print(f"{k}")
                #     if k == "input_pts":
                #         print(len(v), len(v[0]), len(v[1]))
                #     elif k == "depth":
                #         print(v["stage1"].shape)
                #     elif k == "scan":
                #         print(v)
                #     try:
                #         print(v.shape)
                #     except:
                #         print(len(v))

                sample_cuda = tocuda(sample)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
                depth_values = sample_cuda["depth_values"]
                depth_interval = depth_values[:, 1] - depth_values[:, 0]

                self.optimizer.zero_grad()
                
                # print("SAMPLETO CUDA GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                # print("SAMPLETO CUDA GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
                # gradient accumulate
                if self.multi_scale:
                    # if self.multi_ratio_scale_batch_map is not None:
                    #     ratio_idx = dl.dataset.scale2idx[tuple(imgs.shape[3:5])]
                    #     bs = dl.dataset.scale_batch_map[str(ratio_idx)]
                    # else:
                    bs = self.scale_batch_map[str(imgs.shape[3])]
                else:
                    bs = imgs.shape[0]
                iters_to_accumulate = imgs.shape[0] // bs
                total_loss = torch.tensor(0.0, device="cuda")
                total_loss_dict = collections.defaultdict(float)

                for bi in range(iters_to_accumulate):
                    b_start = bi * bs
                    b_end = (bi + 1) * bs
                    cam_params_tmp = {}
                    depth_gt_ms_tmp = {}
                    mask_ms_tmp = {}
                    imgs_tmp = imgs[b_start:b_end]
                    for k in cam_params:
                        cam_params_tmp[k] = cam_params[k][b_start:b_end]
                        depth_gt_ms_tmp[k] = depth_gt_ms[k][b_start:b_end]
                        mask_ms_tmp[k] = mask_ms[k][b_start:b_end]
                    
                    # print("BEFORE PREPARE BNV GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                    # print("BEFORE PREPARE BNV GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
                    if self.rgbd:
                        sensor_data, pointnet_model, neural_map = self.prepare_bnvfusion_input(sample_cuda, b_start, b_end)
                    else:
                        sensor_data, pointnet_model, neural_map = None, None, None

                    # print("AFTER PREPARE BNV GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                    # print("AFTER PREPARE BNV GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
                    if self.fp16:
                        # with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.bf16 else torch.float16):

                            outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end], sensor_data, pointnet_model, neural_map, self.dimensions)
                        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
                    else:

                        outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end], sensor_data, pointnet_model, neural_map, self.dimensions)

                    # print("AFTER THE DINO MODEL GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                    # print("AFTER THE DINO MODEL GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
                    if type(self.depth_type) == list:
                        loss_dict = get_multi_stage_losses(self.loss_arg, self.depth_type, outputs, depth_gt_ms_tmp, mask_ms_tmp,
                                                        depth_interval[b_start:b_end], self.inverse_depth)
                    else:
                        loss_dict = get_loss(self.loss_arg, self.depth_type, outputs, depth_gt_ms_tmp,
                                            mask_ms_tmp, depth_interval[b_start:b_end], self.inverse_depth)

                    loss = torch.tensor(0.0, device="cuda")
                    for key in loss_dict:
                        loss = loss + loss_dict[key] / iters_to_accumulate
                        total_loss_dict[key] = total_loss_dict[key] + loss_dict[key] / iters_to_accumulate
                    total_loss += loss

                    if self.fp16:
                        self.scaler.scale(loss * self.loss_downscale).backward()
                    else:
                        (loss * self.loss_downscale).backward()

                del sensor_data, pointnet_model, neural_map
                
                if self.debug:
                    # DEBUG:scaled grad
                    with torch.no_grad():
                        for group in self.optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                if param.grad.is_sparse:
                                    if param.grad.dtype is torch.float16:
                                        param.grad = param.grad.coalesce()
                                    to_unscale = param.grad._values()
                                else:
                                    to_unscale = param.grad
                                v = to_unscale.clone().abs().max()
                                if torch.isinf(v) or torch.isnan(v):
                                    print('Rank', str(self.rank) + ':', 'INF in', group['layer_name'], 'of step',
                                        global_step, '!!!')
                                scaled_grads[group['layer_name']].append(v.item() / self.scaler.get_scale())

                if self.grad_norm is not None:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm, error_if_nonfinite=False)

                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = int(self.scaler.get_scale())
                    if new_scale == pre_scale:  # 只有scale不变表示优化进行了
                        self.lr_scheduler.step()
                    pre_scale = new_scale
                else:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                global_step = (epoch - 1) * len(dl) + batch_idx
                self.wandb_global_step = global_step

                # forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1000.0 ** 2)
                # print(f"imgs shape:{imgs.shape},, iters_to_accumulate:{iters_to_accumulate}, max_mem: {forward_max_memory_allocated}")


                # no_grad_param = []
                # for name,param in self.model.named_parameters():
                #     if param.grad is None:
                #         no_grad_param.append(name)
                # xx = [n for n, p in self.model.named_parameters() if n.startswith("vit.")]
                # not_in_no_grad_param = [k for k in xx if k not in no_grad_param]
                # not_in_vit = [k for k in no_grad_param if k not in xx]
                # pdb.set_trace()

                if self.rank == 0:
                    desc = f"Epoch: {epoch}/{self.epochs}. " \
                        f"Train. " \
                        f"Scale:({imgs.shape[-2]}x{imgs.shape[-1]}), " \
                        f"Loss: {'%.2f' % total_loss.item()}"
                    # FIXME:Temp codes
                    if "stage4_uncertainty" in loss_dict:
                        desc += f", VarLoss: {'%.2f' % loss_dict['stage4_uncertainty'].item()}"

                    if self.fp16:
                        desc += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    t_loader.set_description(desc)
                    t_loader.refresh()

                if self.debug and batch_idx % 50 == 0 and self.rank == 0:
                    scaled_grads_dict = {}
                    for k in scaled_grads:
                        scaled_grads_dict[k] = np.max(scaled_grads[k])
                    save_scalars(self.writer, 'grads', scaled_grads_dict, self.wandb_global_step)
                    scaled_grads = collections.defaultdict(list)

                if batch_idx % self.log_step == 0 and self.rank == 0:
                    scalar_outputs = {"loss": total_loss.item()}
                    for key in total_loss_dict:
                        scalar_outputs['loss_' + key] = loss_dict[key].item()
                    sample_i = 0 # reference image
                    image_outputs = {"pred_depth": outputs['refined_depth'][sample_i] * mask_ms_tmp[f'stage{num_stage}'][sample_i],
                                    "pred_depth_nomask": outputs['refined_depth'][sample_i],
                                    "conf": outputs['photometric_confidence'][sample_i], 'depth_interval': sample_cuda['depth_interval'][sample_i],
                                    "gt_depth": depth_gt_ms_tmp[f'stage{num_stage}'][sample_i], "ref_img": imgs_tmp[sample_i, 0],
                                    "depths_max": sample_cuda["depths_max"][sample_i], "depths_min": sample_cuda["depths_min"][sample_i]}
                    if self.fp16:
                        scalar_outputs['loss_scale'] = float(self.scaler.get_scale())
                    for i, lr_value in enumerate(self.lr_scheduler.get_last_lr()):
                        scalar_outputs[f'lr_{i}'] = lr_value
                    save_scalars(self.writer, 'train', scalar_outputs, self.wandb_global_step)
                    save_images(self.writer, 'train', image_outputs, self.wandb_global_step)
                    del scalar_outputs, image_outputs

                    # print("1 BATCH GPU MEMORY USAGE:", torch.cuda.memory_allocated()/1e9)  # Current GPU memory usage
                    # print("1 BATCH GPU MEMORY RESERVED:", torch.cuda.memory_reserved()/1e9)  # Total GPU memory reserved
        val_metrics = self._valid_epoch(epoch)

        if self.ddp:
            print(f"Dist group: {dist_group}")
            # dist.barrier(group=dist_group)
            for k in val_metrics:
                dist.all_reduce(val_metrics[k], group=dist_group, async_op=False)
                val_metrics[k] /= dist.get_world_size(dist_group)
                val_metrics[k] = val_metrics[k].item()

        if self.rank == 0:
            save_scalars(self.writer, 'test', val_metrics, epoch)
            print("Global Test avg_test_scalars:", val_metrics)
        else:
            val_metrics = {'useless_for_other_ranks': -1}
        if self.ddp:
            dist.barrier(group=dist_group)

        return val_metrics

    def _valid_epoch(self, epoch, temp=False):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        temp: just for test
        :return: A log that contains information about validation
        """
        self.model.eval()
        seen_scans = set()
        global_val_metrics = {}
        with torch.no_grad():
            for val_data_idx, dl in enumerate(self.valid_data_loader):
                self.valid_metrics.reset()
                for batch_idx, sample in enumerate(tqdm(dl)):
                    sample_cuda = tocuda(sample)
                    depth_gt_ms = sample_cuda["depth"]
                    mask_ms = sample_cuda["mask"]
                    num_stage = 3 if self.config['data_loader'][0]['args'].get('stage3', False) else 4
                    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                    mask = mask_ms["stage{}".format(num_stage)]

                    imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                    depth_values = sample_cuda["depth_values"]
                    depth_interval = depth_values[:, 1] - depth_values[:, 0]

                    if self.rgbd:
                        sensor_data, pointnet_model, neural_map = self.prepare_bnvfusion_input(sample_cuda)
                    else:
                        sensor_data, pointnet_model, neural_map = None, None, None
                    if self.fp16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.bf16 else torch.float16):
                            outputs = self.model.forward(imgs, cam_params, depth_values, sensor_data, pointnet_model, neural_map, self.dimensions)
                    else:
                        outputs = self.model.forward(imgs, cam_params, depth_values, sensor_data, pointnet_model, neural_map, self.dimensions)
                    
                    depth_est = outputs["refined_depth"].detach()
                    if self.config['data_loader'][val_data_idx]['type'] == 'BlendedLoader':
                        scalar_outputs = collections.defaultdict(float)
                        for j in range(depth_interval.shape[0]):
                            di = depth_interval[j].item()
                            scalar_outputs_ = {
                                "abs_depth_thres0-2mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 2]),
                                "abs_depth_thres0-4mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 4]),
                                "abs_depth_thres0-8mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 8]),
                                "abs_depth_thres0-14mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 14]),
                                "thres2mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 2),
                                "thres4mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 4),
                                "thres8mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 8),
                                "thres14mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 14)}
                            for k in scalar_outputs_:
                                scalar_outputs[k] += scalar_outputs_[k]
                        for k in scalar_outputs:
                            scalar_outputs[k] /= depth_interval.shape[0]
                    elif self.config['data_loader'][val_data_idx]['type'] == 'Sk3DLoader':
                        di = depth_interval[0].item()
                        scalar_outputs = {"abs_depth_thres0-2mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 2]),
                                          "abs_depth_thres0-4mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 4]),
                                          "abs_depth_thres0-8mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 8]),
                                          "abs_depth_thres0-14mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 14]),
                                          "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 2),
                                          "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 4),
                                          "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 8),
                                          "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 14)}
                    else:
                        di = depth_interval[0].item() / 2.65
                        scalar_outputs = {"abs_depth_thres0-2mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 2]),
                                          "abs_depth_thres0-4mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 4]),
                                          "abs_depth_thres0-8mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 8]),
                                          "abs_depth_thres0-14mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 14]),
                                          "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 2),
                                          "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 4),
                                          "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 8),
                                          "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 14)}
                    
                    del sensor_data, pointnet_model, neural_map
                    scalar_outputs = tensor2float(scalar_outputs)
                    # image_outputs = {"pred_depth": outputs['refined_depth'][0:1] * mask[0:1], "gt_depth": depth_gt[0:1], "ref_img": imgs[0:1, 0]}
                    image_outputs = {"pred_depth": outputs['refined_depth'][0] * mask[0],
                     "gt_depth": depth_gt[0], "ref_img": imgs[0, 0], 'depth_interval': sample_cuda['depth_interval'][0],
                      "depths_max": sample_cuda["depths_max"][0], "depths_min": sample_cuda["depths_min"][0]}

                    self.valid_metrics.update(scalar_outputs)

                    if self.rank == 0 and val_data_idx == 0:
                        # 每个scan存一张
                        # filenames = sample['filename']
                        # for filename in filenames:
                        #     scan = filename.split('/')[0]
                        #     if scan not in seen_scans:
                        #         seen_scans.add(scan)
                        #         save_images(self.writer, 'test', image_outputs, epoch, fname=scan)
                        save_images(self.writer, 'val', image_outputs, self.wandb_global_step)
                        save_scalars(self.writer, 'val', scalar_outputs, self.wandb_global_step)

                    if temp and batch_idx > 5:
                        break

                val_metrics = self.valid_metrics.mean()
                val_metrics['mean_error'] = val_metrics['thres2mm_error'] + val_metrics['thres4mm_error'] + \
                                            val_metrics['thres8mm_error'] + val_metrics['thres14mm_error']
                val_metrics['mean_error'] = val_metrics['mean_error'] / 4.0

                if len(self.valid_data_loader) > 1 and self.rank == 0:
                    print(f'Eval complete for dataset{val_data_idx}...')
                if val_data_idx > 0:
                    val_metrics_copy = {}
                    for k in val_metrics:
                        val_metrics_copy[f'ex_valset_{val_data_idx}_' + k] = val_metrics[k]
                    val_metrics = val_metrics_copy

                global_val_metrics.update(val_metrics)
                # save_scalars(self.writer, 'test', val_metrics, epoch)
                # print(f"Rank{self.rank}, avg_test_scalars:", val_metrics)

        for k in global_val_metrics:
            global_val_metrics[k] = torch.tensor(global_val_metrics[k], device=self.rank, dtype=torch.float32)
        self.model.train()

        return global_val_metrics

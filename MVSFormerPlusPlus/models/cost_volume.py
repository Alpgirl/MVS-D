import math
from models.warping import homo_warping_3D_with_mask
from models.module import *
# from models.swin.block import SwinTransformerCostReg
import torch.utils.checkpoint as cp
import trimesh
import os
from utils import global_pcd_batch


class GridInterpolation(object):
    def __init__(self, n_xyz, bound_min, voxel_size, n_feats, dtype, target_hw, target_d):
        self.resolution = n_xyz
        self.bound_min = bound_min
        self.voxel_size = voxel_size
        self.n_feats = n_feats
        self.dtype = dtype
        self.target_hw = target_hw
        self.target_d = target_d

        self.full_pp_feats = self.initialize_pp_feat()
        self.full_pp_coord = self.initialize_pp_coord()

    
    def initialize_pp_feat(self):
        return torch.zeros((torch.prod(self.resolution), self.n_feats), dtype=self.dtype, device="cuda")
    
    
    def initialize_pp_coord(self):
        res_x, res_y, res_z = [int(v) for v in self.resolution]

        # represent the full voxel grid as points (N,3)
        xx, yy, zz = torch.meshgrid([torch.arange(res_x, dtype=self.dtype), 
                                     torch.arange(res_y, dtype=self.dtype), 
                                     torch.arange(res_z, dtype=self.dtype)])
        full_pp_coord = torch.stack([xx,yy,zz], axis=-1).reshape(-1,3).cuda()

        # rescale and shift the full voxel grid (from bnv)
        full_pp_coord = full_pp_coord * self.voxel_size + self.bound_min

        # mesh = trimesh.Trimesh(vertices=full_pp_coord.cpu().numpy())
        # output_path = os.path.join(os.getcwd(), "full_pp_coord.ply")
        # mesh.export(output_path)
        # print(f"Mesh exported successfully to {output_path}")
        return full_pp_coord
    

    def reset_pp_grids(self):
        del self.full_pp_coord, self.full_pp_feats
    

    def interpolate_feats(self, bnv_pp_ids, bnv_pp_feats, depth_pp_hyp):
        bnv_pp_ids = bnv_pp_ids.clone().cuda()
        bnv_pp_feats = bnv_pp_feats.clone().to(self.dtype).cuda()
        depth_pp_hyp = depth_pp_hyp.clone().to(self.dtype)

        # flatten act_ids
        bnv_flat_ids = bnv_pp_ids[..., 0] * self.resolution[1] * self.resolution[2] + \
                bnv_pp_ids[..., 1] * self.resolution[2] + bnv_pp_ids[..., 2]

        # fill known features
        self.full_pp_feats[bnv_flat_ids] = bnv_pp_feats

        # reshape and permute features to match the input to grid sample
        full_grid_feats_resh = self.full_pp_feats.reshape(1, self.resolution[0], self.resolution[1], self.resolution[2], self.n_feats) # 1, Nx, Ny, Nz, 3
        full_grid_feats_perm = full_grid_feats_resh.permute(0, 4, 3, 2, 1) # 1, 3, Nz, Ny, Nx

        # normalize the grid to [-1,1] relatively full grid
        mx, my, mz = self.bound_min
        x_max, y_max, z_max = mx + self.resolution[0] * self.voxel_size, my + self.resolution[1] * self.voxel_size, mz + self.resolution[2] * self.voxel_size

        depth_pp_hyp[...,0] = 2 * (depth_pp_hyp[...,0] - mx) / (x_max - mx) - 1
        depth_pp_hyp[...,1] = 2 * (depth_pp_hyp[...,1] - my) / (y_max - my) - 1
        depth_pp_hyp[...,2] = 2 * (depth_pp_hyp[...,2] - mz) / (z_max - mz) - 1

        # reshape hyp_pp to the volume
        depth_grid_hyp = depth_pp_hyp.reshape(1, self.target_d, self.target_hw[0], self.target_hw[1],  3) 

        # interpolate
        with torch.no_grad():
            output = torch.nn.functional.grid_sample(
                full_grid_feats_perm,  # Input volume (N, C, Din, Hin, Win)
                depth_grid_hyp,  # Grid coordinates (N, Dout, Hout, Wout, 3)
                mode='bilinear',  # Interpolation mode
                padding_mode='zeros',  # Padding mode for out-of-bound points
                align_corners=False  # Align corners for consistent interpolation
            )

        return output
    

    def interpolate_coords(self, depth_pp_hyp):
        depth_pp_hyp = depth_pp_hyp.clone().to(self.dtype)

        # reshape and permute features to match the input to grid sample
        full_grid_coord_resh = self.full_pp_coord.reshape(1, self.resolution[0], self.resolution[1], self.resolution[2], 3) # 1, Nx, Ny, Nz, 3
        full_grid_coord_perm = full_grid_coord_resh.permute(0, 4, 3, 2, 1) # 1, 3, Nz, Ny, Nx

        # normalize the grid to [-1,1] relatively full grid
        mx, my, mz = self.bound_min
        x_max, y_max, z_max = mx + self.resolution[0] * self.voxel_size, my + self.resolution[1] * self.voxel_size, mz + self.resolution[2] * self.voxel_size

        depth_pp_hyp[...,0] = 2 * (depth_pp_hyp[...,0] - mx) / (x_max - mx) - 1
        depth_pp_hyp[...,1] = 2 * (depth_pp_hyp[...,1] - my) / (y_max - my) - 1
        depth_pp_hyp[...,2] = 2 * (depth_pp_hyp[...,2] - mz) / (z_max - mz) - 1

        # shift the full grid
        full_grid_coord_perm += self.voxel_size / 2

        # reshape hyp_pp to the volume
        depth_grid_hyp = depth_pp_hyp.reshape(1, self.target_d, self.target_hw[0], self.target_hw[1],  3) 

        # filter such that avoid points near the border (boundary effects)
        is_valid = (depth_grid_hyp <= 0.9).all(-1) & (depth_grid_hyp >= -0.9).all(-1)

        # interpolate
        with torch.no_grad():
            output = torch.nn.functional.grid_sample(
                full_grid_coord_perm,  # Input volume (1, C, Din, Hin, Win)
                depth_grid_hyp[is_valid].unsqueeze(0).unsqueeze(1).unsqueeze(2),  # Grid coordinates (1, 1, 1, N, 3)
                mode='bilinear',  # Interpolation mode
                padding_mode='zeros',  # Padding mode for out-of-bound points
                align_corners=False  # Align corners for consistent interpolation
            )

        # reshape resampled coordinates back to points (N,3)
        output_pp = (output.permute(0, 2, 3, 4, 1)).reshape(-1,3)

        return output_pp



class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


class StageNet(nn.Module):
    def __init__(self, args, bnvconfig, ndepth, stage_idx):
        super(StageNet, self).__init__()
        self.args = args
        self.fusion_type = args.get('fusion_type', 'cnn')
        self.ndepth = ndepth
        self.bnvconfig = bnvconfig
        self.stage_idx = stage_idx
        self.cost_reg_type = args.get("cost_reg_type", ["Normal", "Normal", "Normal", "Normal"])[stage_idx]
        self.depth_type = args["depth_type"]
        if type(self.depth_type) == list:
            self.depth_type = self.depth_type[stage_idx]

        in_channels = args['base_ch']
        if type(in_channels) == list:
            in_channels = in_channels[stage_idx]
        if self.fusion_type == 'cnn':
            self.vis = nn.Sequential(ConvBnReLU(1, 16), ConvBnReLU(16, 16), ConvBnReLU(16, 8), nn.Conv2d(8, 1, 1), nn.Sigmoid())
        else:
            raise NotImplementedError(f"Not implemented fusion type: {self.fusion_type}.")

        if self.cost_reg_type == "PureTransformerCostReg":
            args['transformer_config'][stage_idx]['base_channel'] = in_channels
            self.cost_reg = PureTransformerCostReg(in_channels, **args['transformer_config'][stage_idx])
        else:
            model_th = args.get('model_th', 8)
            if ndepth <= model_th:  # do not downsample in depth range
                self.cost_reg = CostRegNet3D(in_channels, in_channels)
            else:
                self.cost_reg = CostRegNet(in_channels, in_channels)
        self.use_adapter = args.get("use_adapter", False)
        if self.use_adapter:
            in_ch = bnvconfig.model.feature_vector_size+args["feat_chs"][0]
            out_ch = args["feat_chs"][0]
            self.adapter = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            # initialize adapter in the way that identity weight matrix corresponds to rgb feats and small random values correspond to depth feats
            self.adapter.weight.data[:, :out_ch] = torch.eye(out_ch)
            self.adapter.weight.data[:, out_ch:] = torch.normal(mean=0.0, std=0.001, size=(out_ch, abs(in_ch - out_ch)))
            self.adapter.bias.data = torch.ones(out_ch) * 1e-5

    def forward(self, features, proj_matrices, depth_values, depth_features, interpolater, tmp, position3d=None): #dimensions
        ref_feat = features[:, 0]
        src_feats = features[:, 1:] # [1, V-1, 64, 172, 200]
        src_feats = torch.unbind(src_feats, dim=1) # tuple of [1, 64, 172, 200], len=9
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(src_feats) == len(proj_matrices) - 1, "Different number of images and projection matrices"

        # step 1. feature extraction
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # ([1, 2, 4, 4]), ([1, 2, 4, 4],...)-> 9
        # step 2. differentiable homograph, build cost volume
        volume_sum = 0.0
        vis_sum = 0.0
        similarities = []
        with autocast(enabled=False):
            for src_feat, src_proj in zip(src_feats, src_projs):
                # warpped features
                src_feat = src_feat.to(torch.float32) 
                src_proj_new = src_proj[:, 0].clone() # returns a copy of input
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4]) # [B, 3, 3] @ [B, 3, 4]
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume, proj_mask = homo_warping_3D_with_mask(src_feat, src_proj_new, ref_proj_new, depth_values) # [1, 64, D, H, W], [1, D, H, W]

                B, C, D, H, W = warped_volume.shape
                G = self.args['base_ch']
                if type(G) == list:
                    G = G[self.stage_idx]

                if G < C:
                    warped_volume = warped_volume.view(B, G, C // G, D, H, W)
                    ref_volume = ref_feat.view(B, G, C // G, 1, H, W).repeat(1, 1, 1, D, 1, 1).to(torch.float32)
                    in_prod_vol = (ref_volume * warped_volume).mean(dim=2)  # [B,G,D,H,W]
                elif G == C:
                    ref_volume = ref_feat.view(B, G, 1, H, W).to(torch.float32)
                    in_prod_vol = ref_volume * warped_volume  # [B,C(G),D,H,W]
                else:
                    raise AssertionError("G must <= C!")

                if self.fusion_type == 'cnn':
                    sim_vol = in_prod_vol.sum(dim=1)  # [B,D,H,W]
                    sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
                    entropy = (- sim_vol_norm * torch.log(sim_vol_norm + 1e-7)).sum(dim=1, keepdim=True)
                    vis_weight = self.vis(entropy)
                else:
                    raise NotImplementedError

                volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1) # [1, 8, D, 172, 200] + [1, 8, D, 172, 200] * [1, 172, 200]
                vis_sum = vis_sum + vis_weight

            # aggregate multiple feature volumes by variance
            volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6)  # volume_sum / (num_views - 1)

        # VISUALIZE COST VOLUME
        ## DEBUG
        # # convert cost volume to pcd3d
        if depth_features is not None and self.stage_idx == 0:
            points = global_pcd_batch(depth_values, ref_proj) # [b, N, 3]

            inter_pp = []

            for b in range(B):
                pp = interpolater.interpolate_feats(depth_features["active_coordinates"][b],
                                                    depth_features["features"][b],
                                                    points[b,])
                inter_pp.append(pp[0])
            
            bnv_grid_feats = torch.stack(inter_pp, axis=0)
            # print(bnv_grid_feats.shape)

            volume_bnv_mean = torch.cat([volume_mean, bnv_grid_feats], dim=1) # [B, C, D, H, W]
            # print(volume_mean.shape)

            if self.use_adapter:
                volume_mean = self.adapter(volume_bnv_mean.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3) # [B, D, H, W, C]
                # volume_mean += volume_bnv_mean

            del points, pp, inter_pp, bnv_grid_feats
            interpolater.reset_pp_grids()


        # mesh = trimesh.Trimesh(vertices=points[0,].cpu().numpy())
        # output_path = os.path.join(os.getcwd(), "depth_hyp.ply")
        # mesh.export(output_path)
        # print(f"Mesh exported successfully to {output_path}")

        cost_reg = self.cost_reg(volume_mean, position3d) # [1, 1, D, 172, 200]

        prob_volume_pre = cost_reg.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)

        if self.depth_type == 'ce':
            if self.training:
                _, idx = torch.max(prob_volume, dim=1)
                # vanilla argmax
                depth = torch.gather(depth_values, dim=1, index=idx.unsqueeze(1)).squeeze(1)
            else:
                # regression (t)
                depth = depth_regression(F.softmax(prob_volume_pre * tmp, dim=1), depth_values=depth_values)
            # conf
            photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]

        else:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            if self.ndepth >= 32:
                photometric_confidence = conf_regression(prob_volume, n=4)
            elif self.ndepth == 16:
                photometric_confidence = conf_regression(prob_volume, n=3)
            elif self.ndepth == 8:
                photometric_confidence = conf_regression(prob_volume, n=2)
            else:  # D == 4
                photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]

        outputs = {'depth': depth, 'prob_volume': prob_volume, "photometric_confidence": photometric_confidence.detach(),
                   'depth_values': depth_values, 'prob_volume_pre': prob_volume_pre}

        return outputs

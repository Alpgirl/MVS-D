import torch
import torch.distributed as dist
import os
import torch.backends.cudnn as cudnn
# from weakref import KeyedRef
# import hydra
# import numpy as np
# # import os
# from omegaconf import DictConfig
# # import torch
# import time
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from pytorch_lightning import seed_everything
# import trimesh

# from bnv_fusion.src.datasets import datasets
# from bnv_fusion.src.datasets.fusion_inference_dataset import IterableInferenceDataset
# import bnv_fusion.src.utils.o3d_helper as o3d_helper
# import bnv_fusion.src.utils.hydra_utils as hydra_utils
# import bnv_fusion.src.utils.voxel_utils as voxel_utils
# from bnv_fusion.src.models.fusion.local_point_fusion import LitFusionPointNet
# from bnv_fusion.src.models.sparse_volume import SparseVolume
# from bnv_fusion.src.utils.render_utils import calculate_loss
# from bnv_fusion.src.utils.common import to_cuda, Timer
# # import bnv_fusion.third_parties.fusion as fusion
# from numba import njit, prange
# from skimage import measure

# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# FUSION_GPU_MODE = 1

SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = True
cudnn.deterministic = False

def main(rank, local_rank, world_size):
    # Set the GPU device for this process
    torch.cuda.set_device(local_rank)

    # Print information
    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}, GPU: {torch.cuda.current_device()}")

    # Create a tensor on the GPU
    tensor = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")

    # Perform an all_reduce operation (sum)
    # dist.barrier(device_ids=[local_rank])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Print the result
    print(f"Rank {rank} has tensor after all_reduce: {tensor}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    # Get rank, local rank, and world size
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()

    print(f"Local rank {local_rank}, rank {rank}, world size {world_size}")

    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method='tcp://localhost:29500', rank=local_rank, world_size=world_size)

    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'#'INFO'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ["NCCL_DEBUG_SUBSYS"]="COLL"
    # Call the main function with the initialized values
    main(rank, local_rank, world_size)

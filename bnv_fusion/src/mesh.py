import hydra
from weakref import KeyedRef
import hydra
import numpy as np
import os
from omegaconf import DictConfig
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning import seed_everything
import open3d as o3d
import trimesh

import src.utils.hydra_utils as hydra_utils
from src.datasets import datasets
from datasets.fusion_inference_dataset import *


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    if "seed" in config.trainer:
        seed_everything(config.trainer.seed)

    # hydra_utils.extras(config)
    # hydra_utils.print_config(config, resolve=True)
    val_dataset = datasets.get_dataset(config, "val")
    ind = 21
    input_pts = val_dataset[ind][0]["input_pts"][:,:3]
    normals = val_dataset[ind][0]["input_pts"][:,3:]
    print(input_pts.shape)
    

    mesh = trimesh.Trimesh(vertices=input_pts) #, faces=trimesh.convex.convex_hull(input_pts).faces)
    mesh.vertex_normals = normals
    output_path = os.path.join(os.getcwd(), "mesh.ply")
    mesh.export(output_path)
    print(f"Mesh exported successfully to {output_path}")



if __name__ == "__main__":
    main()
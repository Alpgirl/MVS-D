import argparse
import collections
import copy
import pdb

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
# from tensorboardX import SummaryWriter
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets.balanced_sampling import CustomConcatDataset, BalancedRandomSampler
from datasets.sk3d_dataset import Sk3DDataset

from trainer.mvsformer_trainer import Trainer
from base.parse_config import ConfigParser
from utils import get_lr_schedule_with_warmup, get_parameter_groups, init_model, read_json, custom_collate_fn, DotDict

SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = True
cudnn.deterministic = False


def main(local_rank, args, config, bnvconfig):
    # rank = args.node_rank * args.gpus + gpu
    rank = args.rank
    # print(f"rank={rank}, args.rank={args.rank}")
    # print(f"local rank={gpu}")
    if args.DDP:
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank, group_name='mtorch')
        dist.init_process_group(backend='nccl',
                                init_method='tcp://localhost:29500',
                                rank=local_rank,
                                world_size=args.world_size)

        # print('Nodes:', args.nodes, 'Node_rank:', args.node_rank, 'Rank:', rank, 'GPU_id:', gpu)
    torch.cuda.set_device(local_rank)
    # print(f"process {local_rank}  is reaching the barrier!")
    # dist.barrier(device_ids=[local_rank])
    # print(f"process {local_rank}  passed the barrier!")

    train_data_loaders, valid_data_loaders = [], []
    train_sampler = None
    for dl_params in config['data_loader']:
        dl_name, dl_args = dl_params['type'], dict(dl_params['args'])
        train_dl_args = dl_args.copy()
        train_dl_args['listfile'] = dl_args['train_data_list']
        train_dl_args['batch_size'] = train_dl_args['batch_size'] // args.world_size
        train_dl_args['world_size'] = args.world_size

        # set dataname for config
        config['arch']['dataset_name'] = dl_name

        del train_dl_args['train_data_list'], train_dl_args['val_data_list']

        if train_dl_args['multi_scale']:
        #     from datasets.blended_dataset_ms import BlendedMVSDataset
        #     from datasets.dtu_dataset_ms import DTUMVSDataset
            cudnn.benchmark = False  # benchmark=False is more suitable for the multi-scale training
        # else:
        #     from datasets.blended_dataset import BlendedMVSDataset
        #     from datasets.dtu_dataset import DTUMVSDataset

        if args.balanced_training: # False
            train_dl_args_dtu = copy.deepcopy(train_dl_args)
            train_dl_args_dtu['datapath'] = train_dl_args_dtu['dtu_datapath']
            train_dl_args_dtu['listfile'] = train_dl_args_dtu['dtu_train_data_list']
            train_dataset1 = DTUMVSDataset(**train_dl_args_dtu)

            train_dl_args_blended = copy.deepcopy(train_dl_args)
            train_dl_args_blended['datapath'] = train_dl_args_blended['blended_datapath']
            train_dl_args_blended['listfile'] = train_dl_args_blended['blended_train_data_list']
            train_dataset2 = BlendedMVSDataset(**train_dl_args_blended)

            train_dataset = CustomConcatDataset([train_dataset1, train_dataset2])
            train_sampler = BalancedRandomSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        else:
            # if dl_name == 'BlendedLoader':
            #     train_dataset = BlendedMVSDataset(**train_dl_args)
            # else:
            #     train_dataset = DTUMVSDataset(**train_dl_args)

            train_dataset = Sk3DDataset(**train_dl_args)

            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        print(f"train batch size: {train_dl_args['batch_size']}")
        train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=train_dl_args['batch_size'],
                                  num_workers=train_dl_args['num_workers'], sampler=train_sampler, drop_last=True, collate_fn=custom_collate_fn) # train_dl_args['num_workers'] 
        train_data_loaders.append(train_loader)
        
        # setup valid_data_loader instances
        val_kwags = {
            "listfile": dl_args['val_data_list'],
            "mode": "val",
            "nviews": 5,
            "shuffle": False,
            "batch_size": dl_args['eval_batch_size'],
            "random_crop": False
        }
        val_dl_args = train_dl_args.copy()
        val_dl_args.update(val_kwags)

        # if dl_name == 'BlendedLoader':
        #     val_dataset = BlendedMVSDataset(**val_dl_args)
        # else:
        #     val_dataset = DTUMVSDataset(**val_dl_args)

        val_dataset = Sk3DDataset(**val_dl_args)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
        # 根据测试图片尺度评估batchsize
        eval_batch = val_dl_args["batch_size"] // args.world_size #train_dl_args['batch_size']
        # if dl_args['width'] > 1024:
            # eval_batch = 2
        # if dl_args['width'] > 1536:
            # eval_batch = 1
        val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, batch_size=eval_batch, num_workers=4, sampler=val_sampler, collate_fn=custom_collate_fn)
        valid_data_loaders.append(val_data_loader)

        if args.balanced_training: # False
            val_dl_args2 = copy.deepcopy(val_dl_args)
            if dl_name == 'BlendedLoader':
                val_dl_args2['datapath'] = val_dl_args2['dtu_datapath']
                val_dl_args2['listfile'] = val_dl_args2['dtu_val_data_list']
                val_dl_args2['height'] = 1152
                val_dl_args2['width'] = 1536
                val_dataset2 = DTUMVSDataset(**val_dl_args2)
                config['data_loader'].append({"type": "DTULoader"})
            else:
                val_dl_args2['datapath'] = val_dl_args2['blended_datapath']
                val_dl_args2['listfile'] = val_dl_args2['blended_val_data_list']
                val_dl_args2['height'] = 1536
                val_dl_args2['width'] = 2048
                val_dataset2 = BlendedMVSDataset(**val_dl_args2)
                config['data_loader'].append({"type": "BlendedLoader"})
            val_sampler2 = DistributedSampler(val_dataset2, num_replicas=args.world_size, rank=rank, shuffle=False)
            val_data_loader2 = DataLoader(val_dataset2, shuffle=False, pin_memory=True, batch_size=eval_batch, num_workers=4, sampler=val_sampler2)
            valid_data_loaders.append(val_data_loader2)
            break

    # build models architecture, then print to console
    model = init_model(config, bnvconfig)
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    opt_args = config['optimizer']['args']

    # build optimizer with layer-wise lr decay (lrd)

    if not config['arch']['args'].get('freeze_vit', True):
        for k,v in model.named_parameters():
            if k.startswith("vit."):
                v.requires_grad = True
            if k == 'vit.mask_token':
                v.requires_grad = False
    param_groups = get_parameter_groups(opt_args, model, freeze_vit=config['arch']['args'].get('freeze_vit', None))
    optimizer = torch.optim.AdamW(param_groups, lr=opt_args['lr'], weight_decay=opt_args['weight_decay'])
    lr_scheduler = get_lr_schedule_with_warmup(optimizer, num_warmup_steps=opt_args['warmup_steps'], min_lr=opt_args['min_lr'],
                                               total_steps=len(train_data_loaders[0]) * config['trainer']['epochs'])

    if rank == 0:
        writer = wandb.init(project="mvsformerpp+bnvfusion", name=args.exp_name, config=config)  # Initialize wandb
    else:
        writer = None
    # writer = SummaryWriter(config.log_dir)
    model.cuda(local_rank)
    # model.to(device=dist.get_rank())

    is_finetune = config['arch'].get('finetune', False)
    reset_sche = config['arch'].get('reset_sche', True)

    if is_finetune: # False
        restore_path = config['arch']['dtu_model_path']
        checkpoint = torch.load(restore_path, map_location='cpu')
        if rank == 0:
            print('Load Model from', restore_path, 'Rank:', rank, 'Epoch:{}'.format(checkpoint['epoch']))
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if "pe_dict" in k:
                continue
            k_ = k[7:] if k.startswith('module.') else k
            state_dict[k_] = v
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if not reset_sche:
            start_epoch = checkpoint['epoch'] + 1
            print('Start from epoch', start_epoch)
            for _ in tqdm(range(checkpoint['epoch'] * len(train_data_loaders[0])), disable=True if rank != 0 else False):
                lr_scheduler.step()
        else:
            start_epoch = 1
            for pg in optimizer.param_groups:  # reset initial lr
                if 'vit_param' in pg and pg['vit_param']:
                    pg['lr'] = opt_args['vit_lr']
                    pg['initial_lr'] = opt_args['vit_lr']
                else:
                    pg['lr'] = opt_args['lr']
                    pg['initial_lr'] = opt_args['lr']
    else:
        start_epoch = 1

    if args.resume is not None: # None
        checkpoint = torch.load(args.resume, map_location='cpu')
        if rank == 0:
            print('Load Model from', args.resume, 'Rank:', rank, 'Epoch:{}'.format(checkpoint['epoch']))
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            k_ = k[7:] if k.startswith('module.') else k
            state_dict[k_] = v
            # state_dict[k.replace('module.', '')] = v
        model.load_state_dict(state_dict, strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('Start from epoch', start_epoch)
        for _ in tqdm(range(checkpoint['epoch'] * len(train_data_loaders[0])), disable=True if rank != 0 else False):
            lr_scheduler.step()

    # DEBUG TWICE BACKWARD
    # checkpoint = torch.load(config['arch']['args']['model_path'])
    # state_dict = {}
    # for k,v in checkpoint['state_dict'].items():
    #     k_ = k[7:] if k.startswith('module.') else k
    #     state_dict[k_] = v
    # model.load_state_dict(state_dict, strict=True)

    # PRINT MODEL LAYERS
    if rank == 0:
        for i, (k, v) in enumerate(model.named_parameters()):
            # if i in [437]:
            print(f"{k} with shape {v.shape} requires grad {v.requires_grad}")

    if args.DDP: # False
        if rank == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)#, output_device=gpu, find_unused_parameters=True)
            model._set_static_graph()
        except Exception as e:
            print(e)

    trainer = Trainer(model, optimizer, config=config, bnvconfig=bnvconfig, data_loader=train_data_loaders, ddp=args.DDP,
                      valid_data_loader=valid_data_loaders, lr_scheduler=lr_scheduler, writer=writer, rank=rank,
                      train_sampler=train_sampler, debug=args.debug)
    trainer.start_epoch = start_epoch

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-bnvc', '--bnvconfig', default=None, type=str,
                      help='bnv fusion config file path (default: None)')
    args.add_argument('-e', '--exp_name', default=None, type=str)
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--dataloader_type', default=None, type=str, help='BlendedMVS or DTU')
    args.add_argument('--finetune', action='store_true', help='BlendedMVS or DTU')
    args.add_argument('--data_path', default=None, type=str, help='data set root path')
    args.add_argument('--dtu_model_path', default=None, type=str, help='MVS model trained on DTU')
    args.add_argument('--nodes', type=int, default=1, help='how many machines')
    args.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    args.add_argument('--DDP', action='store_true', help='DDP')
    args.add_argument('--balanced_training', action='store_true', help='train with balanced DTU and blendedmvs, '
                                                                       'use the less one to decide the epoch iterations')
    # args.add_argument("--local-rank", type=int, default=0)
    args.add_argument('--debug', action='store_true', help='slow down the training, but can check fp16 overflow')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    import os
    print(args)

    ngpu = torch.cuda.device_count()
    # args.gpus = ngpu
    if args.DDP:
        # print(f"Global rank: {os.environ.get('SLURM_PROCID')}, local rank: {os.environ.get('SLURM_LOCALID')}")
        args.world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
        # # Likewise for RANK and LOCAL_RANK:
        args.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

        # args.gpu = args.rank % torch.cuda.device_count()
        # args.world_size = args.nodes * args.gpus
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '1122'
        # args.rank = int(os.environ["RANK"])
        # args.gpu = int(os.environ["LOCAL_RANK"])
        # args.world_size = int(os.environ["WORLD_SIZE"])
        print(f"Global Rank: {args.rank}, Local Rank: {local_rank}, World Size: {args.world_size}, n_gpu: {torch.cuda.device_count()}")
    else:
        local_rank = 0
        args.rank = 0
        args.world_size = 1
    
    if args.dataloader_type is not None:
        config['data_loader'][0]['type'] = args.dataloader_type
        # if args.dataloader_type == 'BlendedLoader':
        #     config['data_loader'][0]['args']['train_data_list'] = "lists/blended/training_list.txt"
        #     config['data_loader'][0]['args']['val_data_list'] = "lists/blended/validation_list.txt"

            # set data path
    if args.data_path is not None:
        config['data_loader'][0]['args']['datapath'] = args.data_path

    if args.dtu_model_path is not None:
        config['arch']['dtu_model_path'] = args.dtu_model_path

    if args.finetune is True:
        config['arch']['finetune'] = True

    if args.bnvconfig is not None:
        bnvconfig = DotDict(dict(read_json(args.bnvconfig)))
    else:
        bnvconfig = {}

    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'#'INFO'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ["NCCL_DEBUG_SUBSYS"]="COLL"
    # mp.spawn(main, nprocs=args.world_size, args=(args, config, bnvconfig))
    main(local_rank, args, config, bnvconfig)

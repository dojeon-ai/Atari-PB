import argparse
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from src.dataloaders import *
from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.train_utils import set_global_seeds
from src.trainers import build_trainer
from typing import List
from dotmap import DotMap
import torch
import wandb
import numpy as np
import re
import os


def run(args):    
    args = DotMap(args)

    # Hydra
    OmegaConf.register_new_resolver("eval", eval)
    hydra.initialize(version_base=None, config_path=args.config_path) 
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    
    # Reproducibility
    set_global_seeds(seed=cfg.seed)

    # Create local dataset in main process
    _ = build_dataloader(cfg.dataloader)

    os.environ['MASTER_ADDR'] = cfg.master_addr 
    os.environ['MASTER_PORT'] = cfg.master_port
    cfg.world_size = cfg.num_nodes * cfg.num_gpus_per_node
    if cfg.world_size > 1: 
        # https://docs.wandb.ai/guides/track/log/distributed-training
        # setup is required if you initiate a W&B Run in a spawned process:
        wandb.setup()
        mp.spawn(run_worker,
                 nprocs=cfg.num_gpus_per_node, 
                 args=(cfg.num_gpus_per_node, cfg))
    else:
        run_worker(gpu_id=0, num_gpus_per_node=1, cfg=cfg)


def run_worker(gpu_id, num_gpus_per_node, cfg):

    cfg.device = 'cuda:' + str(gpu_id)
    cfg.rank = cfg.rank * num_gpus_per_node + gpu_id
    device = torch.device(cfg.device)   
    
    print("Rank {}, Use {} for training".format(cfg.rank, cfg.device))
    
    if cfg.world_size > 1:
        print('Train with distributed data parallel')
        # Setup config
        cfg.dataloader.distributed = True
        cfg.trainer.distributed = True
        cfg.dataloader.batch_size = cfg.dataloader.batch_size // cfg.world_size
        # Register custom resolvers on each subprocess
        OmegaConf.register_new_resolver("last", lambda lst: lst[-1])
        OmegaConf.register_new_resolver("eval", eval)
        # Distributed process setup
        dist.init_process_group(backend='nccl', 
                                init_method='env://',
                                world_size=cfg.world_size, 
                                rank=cfg.rank)    
    else:
        print('Train without distributed data parallel')
        cfg.dataloader.distributed = False
        cfg.trainer.distributed = False

    torch.set_num_threads(1) 
    train_loader, train_sampler, eval_loader, eval_sampler = build_dataloader(cfg.dataloader)
    
    # env
    if cfg.env.num_train_envs > 0:        
        _, env = build_env(cfg.env)
    else:
        env = None
    
    # logger
    logger = WandbTrainerLogger(cfg)

    # model
    model = build_model(cfg.model)

    # load pretrained
    p_cfg = cfg.pretrained
    ckpt_path = p_cfg.ckpt_path
    load_layers = p_cfg.load_layers
    freeze_layers = p_cfg.freeze_layers
    
    if (len(ckpt_path) > 0) and (len(load_layers) > 0): 
        model = logger.load_model(
            path=ckpt_path, 
            device=device, 
            model=model, 
            load_layers=load_layers
        )
    else:
        print(f'The model is trained from scratch')
    
    for layer_name in freeze_layers:
        module = getattr(model, layer_name)    
        for param in module.parameters():
            param.requires_grad = False
        print(f'{layer_name} is frozen and won\'t be trained.')
        
    trainer = build_trainer(cfg=cfg.trainer,
                            device=device,
                            train_loader=train_loader,
                            train_sampler=train_sampler,
                            eval_loader=eval_loader,
                            eval_sampler=eval_sampler,
                            eval_env=env,
                            logger=logger,
                            model=model)

    # train
    trainer.train()
    wandb.finish()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,    default='./configs')
    parser.add_argument('--config_name', type=str,    default='pretrain_bc') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))

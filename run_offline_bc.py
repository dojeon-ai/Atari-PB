import argparse
import torch
import wandb
import numpy as np
import itertools
import re, os
import hydra
import time
from hydra import compose, initialize
import multiprocessing as mp
from src.dataloaders import *
from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.train_utils import set_global_seeds
from src.trainers import build_trainer
from typing import List
from dotmap import DotMap


def run(args):    
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # create configurations for seed * games
    games = cfg.dataloader.games
    seeds = cfg.seeds
    cfg_list = []
    for seed, game in itertools.product(*[seeds, games]):
        _cfg = copy.deepcopy(cfg)
        _cfg.dataloader.ataripb_subdir_name = cfg.dataloader.ataripb_subdir_name + '/' + game
        _cfg.dataloader.games = [game]
        _cfg.env.game = game
        _cfg.seed = seed
        cfg_list.append(_cfg)
        
    # run parallel experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    context = mp.get_context('spawn')
    available_gpus = list(range(cfg.num_gpus_per_node))
    process_dict = {gpu_id: [] for gpu_id in available_gpus}
    # https://docs.wandb.ai/guides/track/log/distributed-training
    wandb.setup()
    
    for cfg in cfg_list:
        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process, status in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove((process,status))
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
                    elif status.value != 0:
                        print(f"Process {process.pid} on GPU {gpu_id} has failed!")
                        process.terminate()
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < cfg.num_exps_per_gpu:
                    wait = False
                    gpu_id, processes = min(process_dict.items(), key=lambda x: len(x[1]))
                    break
            time.sleep(1)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        cfg.device = 'cuda:' + str(gpu_id)
        
        # run_worker(cfg, -1)
        process_status = mp.Value('i',0)
        process = mp.Process(target=run_worker, args=(cfg,process_status))
        process.start()
        processes.append((process,process_status))
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == cfg.num_exps_per_gpu:
            available_gpus.remove(gpu_id)

    # wait until all subprocesses finish
    while process_dict:
        finished_gpus = []
        for gpu_id, processes in process_dict.items():
            for process, status in processes:
                if not process.is_alive():
                    print(f"Process {process.pid} on GPU {gpu_id} finished.")
                    processes.remove((process,status))
                elif status.value != 0:
                    print(f"Process {process.pid} on GPU {gpu_id} has failed!")
                    process.terminate()
            if len(processes) == 0:
                finished_gpus.append(gpu_id)

        for gpu_id in finished_gpus:
            process_dict.pop(gpu_id)

        time.sleep(1)


def run_worker(cfg, status):  
    
    try:
        OmegaConf.register_new_resolver("last", lambda lst: lst[-1], replace=True)
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        set_global_seeds(seed=cfg.seed)
        device = torch.device(cfg.device)
        
        # dataloader
        torch.set_num_threads(1) 
        train_loader, train_sampler, eval_loader, eval_sampler = build_dataloader(cfg.dataloader)
        
        # env
        _, env = build_env(cfg.env)
        
        # logger
        logger= WandbTrainerLogger(cfg)

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

                
            print(f'The %s layer is frozen'%(layer_name))

        # trainer
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
        env.close()

    except Exception as e:
        f = open("offline_bc_error.txt",'a')
        f.write(f"Killed: {cfg.env.game} (Seed {cfg.seed})\n")
        f.write(repr(e))
        f.write('\n==============================================\n')
        f.close()
        status.value = -1 # Set failure flag. Some subprocesses won't finish, so the main process should manually terminate those.
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,    default='./configs')
    parser.add_argument('--config_name', type=str,    default='offline_bc') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))

             
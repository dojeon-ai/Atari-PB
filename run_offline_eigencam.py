import argparse
import torch
import wandb
import numpy as np
import itertools
import os
import time
from hydra import compose, initialize
import multiprocessing as mp
from src.dataloaders import *
from src.envs import *
from src.models import *
from src.common.logger import WandbTrainerLogger
from src.common.train_utils import set_global_seeds
from typing import List
from dotmap import DotMap
from src.common.vis_utils import Eigen_CAM, visualize_multi_images
from einops import rearrange
from tqdm import tqdm
from src.envs.atari import ATARI_HUMAN_SCORE


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
        _cfg.dataloader.ataripb_subdir_name = 'eigencam' + '/' + game
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
        
        process_status = mp.Value('i',0)
        run_worker(cfg, process_status)
        # process = mp.Process(target=run_worker, args=(cfg,process_status))
        # process.start()
        # processes.append((process,process_status))
        # print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == cfg.num_exps_per_gpu:
            available_gpus.remove(gpu_id)


def run_worker(cfg, status):  
    
    try:
        set_global_seeds(seed=cfg.seed)
        device = torch.device(cfg.device)
        
        # dataloader
        torch.set_num_threads(1) 
        train_loader, _, _, _ = build_dataloader(cfg.dataloader)
        
        # logger
        _cfg = copy.deepcopy(cfg)
        _cfg.model = cfg.model['CURL']
        _cfg.pretrained = cfg.pretrained['CURL']
        logger= WandbTrainerLogger(_cfg)
        if cfg.use_all:
            if cfg.use_blank:
                model_types={
                    'Input': ['Input'],
                    'Image': ['Random', 'MAE', 'CURL'],
                    'Video': ['ATC', 'SiamMAE', 'R3M'],
                    'Blank': ['Blank'],
                    'Demonstration': ['BC', 'SPR', 'IDM'],
                    'Trajectory': ['DT', 'CQL_M', 'CQL_D']
                }
            else:
                model_types={
                    'Image': ['Input', 'MAE', 'CURL'],
                    'Video': ['ATC', 'SiamMAE', 'R3M'],
                    'Demonstration': ['BC', 'SPR', 'IDM'],
                    'Trajectory': ['DT', 'CQL_M', 'CQL_D']
                }
            
            # model
            models = {}
            for type, model_configs in cfg.model.items():
                if type=='CURL':
                    models['ATC']=build_model(model_configs).to(device)
                    models['R3M']=build_model(model_configs).to(device)
                    models['SPR']=build_model(model_configs).to(device)
                elif type=='BC':
                    models['CQL_M']=build_model(model_configs).to(device)
                    models['Random']=build_model(model_configs).to(device)
                models[f'{type}']=build_model(model_configs).to(device)
                
        else:
            model_types={
                    'Input': ['Input'],
                    'Image': ['CURL'],
                    'Video': ['ATC'],
                    'Demonstration': ['BC'],
                    'Trajectory': ['CQL_D']
                }
            
            # model
            models = {}
            for type, model_configs in cfg.model.items():
                if type=='CURL':
                    models['ATC']=build_model(model_configs).to(device)
                models[f'{type}']=build_model(model_configs).to(device)
        
        # load pretrained
        p_cfg = cfg.pretrained
        ckpt_path_dict = {model_type: pre_cfg.ckpt_path for model_type, pre_cfg in p_cfg.items()}
        load_layers_dict = {model_type: pre_cfg.load_layers for model_type, pre_cfg in p_cfg.items()}
        
        # trainers={}
        cams={}
        print(f'Loading model start!')
        for model_type in tqdm(models.keys()):
            if (len(ckpt_path_dict[f'{model_type}']) > 0) and (len(load_layers_dict[f'{model_type}']) > 0): 
                models[f'{model_type}'] = logger.load_model(
                    path=ckpt_path_dict[f'{model_type}'], 
                    device=device, 
                    model=models[f'{model_type}'], 
                    load_layers=load_layers_dict[f'{model_type}']
                )
            else:
                print(f'The model is trained from scratch')
            
            cams[f'{model_type}'] = Eigen_CAM(model=models[f'{model_type}'].backbone,
                                              target_layers=models[f'{model_type}'].backbone.layer4)
        
        local_save_path = cfg.local_save_path + '/' + cfg.dataloader.games[0]
        os.makedirs(local_save_path, exist_ok=True)
        
        if cfg.save_type == 'image':
            # get grad_cam
            for i, batch in enumerate(train_loader):
                if i+1 == cfg.dataloader.num_of_samples:
                    break
                
                x = batch['observation'].to(device)
                x[x==104]=0
                
                x = x / 255.0
                if cfg.dataloader.games[0] in ATARI_HUMAN_SCORE.keys():
                    x = rearrange(x, 'n f c h w -> n 1 f c h w')
                    # original_img = batch['observation'][0][-1][:, :, 5:].repeat(3,1,1).cpu().numpy()
                    original_img = batch['observation'][0][-1].repeat(3,1,1).cpu().numpy()
                else:
                    x = rearrange(x, 'n f c1 c2 h w -> n 1 f (c1 c2) h w')
                    # import pdb; pdb.set_trace()
                    img = rearrange(batch['observation'], 'n f c1 c2 h w -> n f (c1 c2) h w')
                    original_img = img[0][-1].repeat(3,1,1).cpu().numpy()
                    original_img[original_img==104]=0
                
                # grad cam
                cam_image_list=[]
                cam_image_list.append(original_img.transpose(1,2,0))
                
                for data_type, model_type_list in model_types.items():
                    # save_path = local_save_path + '/' + str(data_type)
                    for model_type in model_type_list:
                        if cfg.use_blank:
                            if model_type == 'Input':
                                continue
                            elif model_type == 'Blank':
                                blank_img = np.ones_like(original_img.transpose(1,2,0)) * 255
                                cam_image_list.append(blank_img)
                            else:
                                models[f'{model_type}'].eval()
                                cam_image = cams[f'{model_type}'].get_eigencam_image(x)
                                # import pdb; pdb.set_trace()
                                # cam_image = cam_image[:, :, 5:]
                                cam_image_list.append(cam_image.transpose(1,2,0))
                                
                        else:
                            if model_type == 'Input':
                                continue
                            else:
                                models[f'{model_type}'].eval()
                                cam_image = cams[f'{model_type}'].get_eigencam_image(x)
                                # import pdb; pdb.set_trace()
                                # cam_image = cam_image[:, :, 5:]
                                cam_image_list.append(cam_image.transpose(1,2,0))
                        
                        
                # save cam image
                local_save_path = cfg.local_save_path + '/' + cfg.dataloader.games[0] + '/' + cfg.save_file_extension
                os.makedirs(local_save_path, exist_ok=True)
                save_path = local_save_path + '/'+ str(i)+ '.'+ cfg.save_file_extension
                
                if cfg.use_all:
                    if cfg.use_blank:
                        visualize_multi_images(image_list=cam_image_list,
                                                model_types=model_types,
                                                rows=2,
                                                cols=7,
                                                mode=cfg.mode,
                                                save_path=save_path)
                    else:
                        visualize_multi_images(image_list=cam_image_list,
                                                model_types=model_types,
                                                rows=2,
                                                cols=6,
                                                mode=cfg.mode,
                                                save_path=save_path)
                else:
                    visualize_multi_images(image_list=cam_image_list,
                                                model_types=model_types,
                                                rows=1,
                                                cols=5,
                                                mode=cfg.mode,
                                                save_path=save_path)
                # import pdb; pdb.set_trace()
        
        elif cfg.save_type == 'video':
            for data_type, model_type_list in model_types.items():
                for model_type in model_type_list:
                    cam_frames = []
                    print(f'{model_type} Start!')
                    for i, batch in tqdm(enumerate(train_loader)):
                        if model_type == 'Input':
                            cam_frames.append(batch['observation'][0][-1].repeat(3,1,1).cpu().numpy())
                            
                        elif model_type == "Blank":
                            continue
                            
                        else:
                            x = batch['observation'].to(device)
                            x = x / 255.0
                            x = rearrange(x, 'n f c h w -> n 1 f c h w')
                            
                            models[f'{model_type}'].eval()
                            cam_image = cams[f'{model_type}'].get_eigencam_image(x)
                            cam_frames.append(cam_image)
                    
                    cam_video = np.array(cam_frames)
                    cam_video = wandb.Video(cam_video, fps=12, format='gif')
                    wandb.log({f'{model_type}': cam_video})
            
            # import pdb; pdb.set_trace()
            # cams[f'{model_type}'].save_eigencam_image(cam_image_list, save_path, f'{i}.png')        
            
        wandb.finish()
        # env.close()

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
    parser.add_argument('--config_name', type=str,    default='offline_eigencam') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))

             
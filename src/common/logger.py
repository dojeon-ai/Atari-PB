import wandb
import torch
import os
import json
from omegaconf import OmegaConf
from src.envs.atari import *
from src.common.class_utils import save__init__args
from collections import deque
import numpy as np


class WandbTrainerLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        
        self.run_id = None
        self.rank = cfg.rank
        if cfg.trainer.resume:
            assert len(cfg.trainer.run_ids)==cfg.world_size
            self.run_id = cfg.trainer.run_ids[cfg.rank]
        else:
            self.run_id = wandb.util.generate_id()
            print(f"Rank {self.rank}, running with wandb run_id {self.run_id}")

        wandb.init(project=self.cfg.project_name, 
                   entity=self.cfg.entity,
                   config=dict_cfg,
                   group=self.cfg.group_name,
                   name=self.cfg.wandb_run_name,
                   id=self.run_id,
                   resume='allow',
                   reinit=True,)

        self.logger = TrainerLogger()

    def update_log(self, **kwargs):
        self.logger.update_log(**kwargs)
    
    def write_log(self, step):
        log_data = self.logger.fetch_log()
        wandb.log(log_data, step=step)

    def save_dict(self, save_dict, name):
        # save checkpoint with premade dict
        path = f'./materials/model/{self.cfg.project_name}/{self.cfg.group_name}/{self.cfg.exp_name}_{self.run_id}/{name}.pth'
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
            
        torch.save(save_dict, path)
        print("Checkpoint saved successfully at", path)

    def load_dict(self, load_dict, path, device):
        # load_dict: keys correspond to each key in the checkpoint loaded from 'path'; values are the loading targets.
        return_dict = {}
        print("Loading checkpoint from", path)
        try:
            checkpoint = torch.load(path, map_location=device)
            for name, target in load_dict.items():
                if hasattr(target, 'load_state_dict'):
                    target.load_state_dict(checkpoint[name])
                else:
                    return_dict[name] = checkpoint[name]
            print("Checkpoint loaded successfully from", path)
        except Exception as e:
            print(e)
            raise KeyError("Loading failed.")

        return return_dict
    
    def load_model(self, path, device, model, load_layers):
        print("Loading model checkpoint")
        try:
            checkpoint = torch.load(path, map_location=device)
            # check whether the pretrained model is wrapped with DDP module
            ex_key = list(checkpoint['backbone'].keys())[0]
            
            def fix_state_dict(state_dict, key):
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace(key, '') 
                    new_state_dict[name] = v
                return new_state_dict
            
            if '_orig_mod.module.' in ex_key:                
                checkpoint['backbone'] = fix_state_dict(checkpoint['backbone'], '_orig_mod.module.')
                checkpoint['neck'] = fix_state_dict(checkpoint['neck'], '_orig_mod.module.')
                checkpoint['head'] = fix_state_dict(checkpoint['head'], '_orig_mod.module.')
                
            elif '_orig_mod.' in ex_key:                
                checkpoint['backbone'] = fix_state_dict(checkpoint['backbone'], '_orig_mod.')
                checkpoint['neck'] = fix_state_dict(checkpoint['neck'], '_orig_mod.')
                checkpoint['head'] = fix_state_dict(checkpoint['head'], '_orig_mod.')
                
            for layer_name in load_layers:
                print('[Load: %s layers]' %(layer_name))
                module = getattr(model, layer_name)
                module.load_state_dict(checkpoint[layer_name])
                print('[All keys matched]')
            print("Loading successful")

        except Exception as e:
            print(e)
            raise KeyError("Loading failed.")
        
        return model


class TrainerLogger(object):
    def __init__(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_set.update(k, v)
            else:
                self.media_set[k] = v

    def fetch_log(self):
        log_data = {}
        log_data.update(self.average_meter_set.averages())
        log_data.update(self.media_set)
        self.reset()
        
        return log_data

    def reset(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}



class WandbAgentLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.run_id = wandb.util.generate_id()

        wandb.init(project=cfg.project_name, 
                   entity=cfg.entity,
                   config=dict_cfg,
                   group=self.cfg.group_name,
                   name=self.cfg.wandb_run_name,
                   id=self.run_id,
                   reinit=True,
                   settings=wandb.Settings(start_method="fork")) # thread    

        self.train_logger = AgentLogger(
            num_envs=cfg.env.num_train_envs,
            env_type=cfg.env.type, 
            game=cfg.env.game,
        )
        self.eval_logger = AgentLogger(
            num_envs=cfg.env.num_eval_envs,
            env_type=cfg.env.type, 
            game=cfg.env.game,
        )
        self.timestep = 0
    
    def step(self, state, reward, done, info, mode='train'):
        if mode == 'train':
            self.train_logger.step(state, reward, done, info)
            self.timestep += 1

        elif mode == 'eval':
            self.eval_logger.step(state, reward, done, info)
            
    def is_traj_done(self, mode='train'):
        if mode == 'train':
            return self.train_logger.is_traj_done
        
        elif mode == 'eval':
            return self.eval_logger.is_traj_done

    def update_log(self, mode='train', **kwargs):
        if mode == 'train':
            self.train_logger.update_log(**kwargs)

        elif mode == 'eval':
            self.eval_logger.update_log(**kwargs)
    
    def write_log(self, mode='train'):
        if mode == 'train':
            log_data = self.train_logger.fetch_log(mode)

        elif mode == 'eval':
            log_data = self.eval_logger.fetch_log(mode)

        # prefix
        log_data = {mode+'_'+k: v for k, v in log_data.items() }
        wandb.log(log_data, step=self.timestep)
        
    def save_dict(self, save_dict, name):
        # save checkpoint with premade dict
        path = f'./materials/model/{self.cfg.project_name}/{self.cfg.group_name}/{self.cfg.exp_name}_{self.run_id}/{name}.pth'
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
            
        torch.save(save_dict, path)
        print("Checkpoint saved successfully at", path)

    def load_dict(self, load_dict, path, device):
        # load_dict: keys correspond to each key in the checkpoint loaded from 'path'; 
        # values are the loading targets.
        return_dict = {}
        print("Loading checkpoint from", path)
        try:
            checkpoint = torch.load(path, map_location=device)
            for name, target in load_dict.items():
                if hasattr(target, 'load_state_dict'):
                    target.load_state_dict(checkpoint[name])
                else:
                    return_dict[name] = checkpoint[name]
            print("Checkpoint loaded successfully from", path)
        except Exception as e:
            print(e)
            raise KeyError("Loading failed.")

        return return_dict
    
    def load_model(self, path, device, model, load_layers):
        print("Loading model checkpoint")
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # check whether the pretrained model is wrapped with DDP module
            ex_key = list(checkpoint['backbone'].keys())[0]
            if '_orig_mod.module' in ex_key or '_orig_mod.' in ex_key:
                def fix_state_dict(state_dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace('_orig_mod.module.', '') 
                        name = name.replace('_orig_mod.', '')
                        new_state_dict[name] = v
                    return new_state_dict
                
                checkpoint['backbone'] = fix_state_dict(checkpoint['backbone'])
                checkpoint['neck'] = fix_state_dict(checkpoint['neck'])
                checkpoint['head'] = fix_state_dict(checkpoint['head'])

            for layer_name in load_layers:
                print('[Load: %s layers]' %(layer_name))
                module = getattr(model, layer_name)
                module.load_state_dict(checkpoint[layer_name])
                print('[All keys matched]')
            
            print("Loading successful")
            
        except Exception as e:
            print(e)
            raise KeyError("Loading failed.")
        
        return model
    
    
class AgentLogger(object):
    def __init__(self, num_envs=100, env_type='atari', game=None):
        # https://arxiv.org/pdf/1709.06009.pdf 
        # Section 3.1 -> Training: end-of-life / Evaluation: end-of-trajectory
        # episode = life / traj = all lives
        self.num_envs = num_envs
        self.env_type = env_type
        self.game = game
        
        self.reset()
        
    def step(self, states, rewards, dones, infos):
        if self.num_envs == 1:
            states = [states]
            rewards = [rewards]
            dones = [dones]
            infos = [infos]
            
        for idx in range(self.num_envs):
            reward = rewards[idx]
            done = dones[idx]
            info = infos[idx]
                
            self.traj_rewards[idx].append(reward)
            self.traj_game_scores[idx].append(info.game_score)

            if info.traj_done:
                self.traj_rewards_buffer[idx].append(np.sum(self.traj_rewards[idx]))
                self.traj_game_scores_buffer[idx].append(np.sum(self.traj_game_scores[idx]))
                self.traj_rewards[idx] = []
                self.traj_game_scores[idx] = []
                    
    @property
    def is_traj_done(self):
        # check whether all the trajectories within the environments are finished
        if all(buffer for buffer in self.traj_game_scores_buffer):
            return True
        else:
            return False
    
    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_set.update(k, v)
            else:
                self.media_set[k] = v

    def fetch_log(self, mode):
        log_data = {}
        
        log_data.update(self.average_meter_set.averages())
        log_data.update(self.media_set)
        self._reset_meter_set()
        
        if self.is_traj_done:
            if mode == 'train':
                agent_reward = np.mean(self.traj_rewards_buffer)
                agent_score = np.mean(self.traj_game_scores_buffer)
                
            elif mode == 'eval':
                agent_reward = sum(lst[0] for lst in self.traj_rewards_buffer) / self.num_envs
                agent_score = sum(lst[0] for lst in self.traj_game_scores_buffer) / self.num_envs            

            log_data['mean_traj_rewards'] = agent_reward
            log_data['mean_traj_game_scores'] = agent_score
            
            if self.env_type == 'atari':
                if self.game in ATARI_RANDOM_SCORE:
                    random_score = ATARI_RANDOM_SCORE[self.game]
                    human_score = ATARI_HUMAN_SCORE[self.game]
                    dqn_score = ATARI_DQN50M_SCORE[self.game]
                    
                elif self.game in FAR_OOD_RANDOM_SCORE:
                    random_score = FAR_OOD_RANDOM_SCORE[self.game]
                    human_score = FAR_OOD_HUMAN_SCORE[self.game]
                    dqn_score = FAR_OOD_RAINBOW_SCORE[self.game]
                
                else:
                    raise NotImplementedError
                            
                hns = (agent_score - random_score) / (human_score - random_score + 1e-6)
                # for dqn normalized score, we follow the protocol from Agarwal et al.
                # the max is needed since DQN performs worse than a random agent on the few games
                # https://arxiv.org/pdf/1907.04543.pdf
                min_score = min(random_score, dqn_score)
                max_score = max(random_score, dqn_score)
                dns = (agent_score - min_score) / (max_score - min_score + 1e-6)
                
                log_data['human_normalized_score'] = hns
                log_data['dqn_normalized_score'] = dns
            
            # if mode is train, trajectory rewards should be tracked after logging
            if mode == 'train':
                self._reset_buffer()
            elif mode == 'eval':
                self._reset_list()
                self._reset_buffer()

        return log_data
    
    def _reset_list(self):
        self.traj_rewards = []
        self.traj_game_scores = []

        for _ in range(self.num_envs):
            self.traj_rewards.append([])
            self.traj_game_scores.append([])       
            
    def _reset_buffer(self):
        self.traj_rewards_buffer = []
        self.traj_game_scores_buffer = []

        for _ in range(self.num_envs):
            self.traj_rewards_buffer.append([])
            self.traj_game_scores_buffer.append([])
            
    def _reset_meter_set(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}
        
    def reset(self):
        self._reset_list()
        self._reset_buffer()
        self._reset_meter_set()
    

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
import copy
from .base import BaseEnv
from .atari import AtariEnv
from .vec_env import VecEnv
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses

ENVS = {subclass.get_name():subclass
        for subclass in all_subclasses(BaseEnv)}

def build_env(cfg):     
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    env_type = cfg.pop('type')
    env = ENVS[env_type]  
    
    num_train_envs = cfg.pop('num_train_envs')
    num_eval_envs = cfg.pop('num_eval_envs')
    
    # build training env
    if num_train_envs > 1:
        raise NotImplementedError('For training, only 1 train_env is supported')
    train_env = env(**cfg)
    
    # build evaluation env
    eval_envs = []
    for idx in range(num_eval_envs):
        _cfg = copy.deepcopy(cfg)
        _cfg['seed'] = cfg['seed'] + idx
        eval_env = env(**_cfg)
        eval_envs.append(eval_env)

    eval_env = VecEnv(num_processes = num_eval_envs, envs = eval_envs)
    
    return train_env, eval_env

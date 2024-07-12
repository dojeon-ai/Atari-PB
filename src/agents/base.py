import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import copy
import numpy as np
import wandb
import matplotlib.pyplot as plt

from collections import deque
from einops import rearrange
from abc import *
from typing import Tuple

from src.common.schedulers import LinearScheduler, ExponentialScheduler, CosineScheduler
from src.common.metrics import explained_variance
from src.common.vis_utils import visualize_histogram


class BaseAgent(metaclass=ABCMeta):
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__()  
        self.cfg = cfg  
        self.device = device
        self.train_env = train_env
        self.eval_env = eval_env
        self.game_id = self.train_env.game_id
        
        self.logger = logger
        self.buffer = buffer
        self.aug_func = aug_func.to(self.device)
        self.model = model.to(self.device)
        self.optimizer_type = cfg.optimizer['type']
        self.optimizer = self._build_optimizer(self.model.parameters(), cfg.optimizer)
        
        self.prior_weight_scheduler = self._build_scheduler(cfg.prior_weight_scheduler)
        self.eps_scheduler = self._build_scheduler(cfg.eps_scheduler)
        self.gamma_scheduler = self._build_scheduler(cfg.gamma_scheduler)
        self.n_step_scheduler = self._build_scheduler(cfg.n_step_scheduler)
        

    @classmethod
    def get_name(cls):
        return cls.name

    def _build_optimizer(self, param_group, optimizer_cfg):
        if 'type' in optimizer_cfg:
            self.optimizer_type = optimizer_cfg.pop('type')
        if self.optimizer_type == 'adam':
            return optim.Adam(param_group, 
                              **optimizer_cfg)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(param_group, 
                              **optimizer_cfg)
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(param_group, 
                                 **optimizer_cfg)
        else:
            raise ValueError
        
    def _build_scheduler(self, scheduler_cfg):
        scheduler_type = scheduler_cfg.pop('type')
        if scheduler_type == 'linear':
            return LinearScheduler(**scheduler_cfg)
        if scheduler_type == 'exponential':
            return ExponentialScheduler(**scheduler_cfg)
        if scheduler_type == 'cosine':
            return CosineScheduler(**scheduler_cfg)
        else:
            raise ValueError
    
    @abstractmethod
    def predict(self, obs, eps) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [output] loss
        [output] pred: prediction Q-value
        [output] target: target Q-value
        """
        pass
    
    def train(self):
        obs = self.train_env.reset()
        self.initial_model = copy.deepcopy(self.model)
        online_model = self.model
        target_model = self.target_model
        
        if self.cfg.exploration_model == 'online':
            exploration_model = self.model
        else:
            exploration_model = self.target_model        

        optimize_step = 1
        self.eps = 1.0
        for env_step in tqdm.tqdm(range(1, self.cfg.num_timesteps+1)):
            ####################
            # collect trajectory
            for module, mode in self.cfg.exploration_mode.items():
                if mode == 'train':
                    getattr(exploration_model, module).train()
                else:
                    getattr(exploration_model, module).eval()
            
            obs_tensor = self.buffer.encode_obs(obs, prediction=True)
            action = self.predict(exploration_model, obs_tensor, self.eps)
            next_obs, reward, done, info = self.train_env.step(action.item())
            self.buffer.store(obs, action, reward, done)
            self.logger.step(obs, reward, done, info, mode='train')

            if info.traj_done:
                obs = self.train_env.reset()
            else:
                obs = next_obs

            if env_step >= self.cfg.min_buffer_size:
                self.eps = self.eps_scheduler.get_value(env_step - self.cfg.min_buffer_size)

                ###################
                # optimize
                for _ in range(self.cfg.optimize_per_env_step):
                    #########################
                    # set train or eval mode
                    for module, mode in self.cfg.train_online_mode.items():
                        if mode == 'train':
                            getattr(online_model, module).train()
                        else:
                            getattr(online_model, module).eval()
                            
                    for module, mode in self.cfg.train_target_mode.items():
                        if mode == 'train':
                            getattr(target_model, module).train()
                        else:
                            getattr(target_model, module).eval()

                    #########################
                    # scheduler
                    # priority is scheduled based on the total timesteps
                    self.prior_weight = self.prior_weight_scheduler.get_value(optimize_step)
                    
                    # gamma & n_step is scheduled based on the reset schedule
                    optimize_step_after_reset = optimize_step % self.cfg.reset_per_optimize_step
                    self.gamma = self.gamma_scheduler.get_value(optimize_step_after_reset)
                    self.n_step = self.n_step_scheduler.get_value(optimize_step_after_reset)
                    self.n_step = int(np.round(self.n_step))
                    
                    #########################
                    # sample batch
                    batch = self.buffer.sample(self.cfg.batch_size, self.n_step, self.gamma, self.prior_weight)
                    
                    #########################
                    # Model Update
                    loss, preds, targets = self.forward(online_model, target_model, batch, mode='train')
                    
                    # optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.cfg.clip_grad_norm
                    )
                    self.optimizer.step()  

                    # target_update
                    # update the buffer information if needed (e.g., running stats from bn),
                    # Do not update freezed layer to reduce computatinoal cost
                    update_layers = ['backbone', 'neck', 'head']
                    update_layers = list(set(update_layers) - set(self.cfg.freeze_layers))
                    for layer in update_layers:
                        online_layer = getattr(online_model, layer)
                        target_layer = getattr(target_model, layer)
                    
                        tau = self.cfg.target_tau
                        for online, target in zip(online_layer.parameters(), target_layer.parameters()):
                            target.data = tau * target.data + (1 - tau) * online.data

                        if self.cfg.update_buffer:
                            for online, target in zip(online_layer.buffers(), target_layer.buffers()):
                                target.data = online.data                            
                    
                    train_logs = {
                        'eps': self.eps,
                        'prior_weight': self.prior_weight,
                        'gamma': self.gamma,
                        'n_step': self.n_step,
                        'loss': loss.item(),
                    }
                    self.logger.update_log(mode='train', **train_logs)
                    optimize_step += 1
                
                ################
                # evaluate
                online_model.eval()
                
                # evaluate
                if (env_step % self.cfg.evaluate_freq == 0) and (self.cfg.evaluate_freq != -1):
                    eval_logs = self.evaluate()
                    self.logger.update_log(mode='eval', **eval_logs)

                if (env_step % self.cfg.rollout_freq == 0) and (self.cfg.rollout_freq != -1):
                    rollout_logs = self.rollout()
                    self.logger.update_log(mode='eval', **rollout_logs)

                ################
                # log
                if env_step % self.cfg.log_freq == 0:
                    self.logger.write_log(mode='train')
                    self.logger.write_log(mode='eval')

                if env_step % self.cfg.save_buffer_every == 0:
                    game_name = self.train_env._game
                    game_name = ''.join(word.capitalize() for word in game_name.split('_'))
                    self.buffer.save_buffer(
                        buffer_dir = self.cfg.buffer_dir,
                        game= game_name,
                        run=self.cfg.run,
                    )
                    
    def evaluate(self):
        ######################
        # Forward
        online_model = self.model
        target_model = self.target_model
        
        # get output of each sublayer with hook
        layer_wise_outputs = {}
        def save_outputs_hook(layer_id):
            def hook(_, __, output):
                layer_wise_outputs[layer_id] = output
            return hook
        
        def register_hooks(net, prefix=''):
            for name, layer in net._modules.items():
                if len(list(layer.children())) > 0:  # Check if it has children
                    register_hooks(layer, prefix=f"{prefix}.{name}")
                else:
                    layer_id = f"{prefix}.{name}.{layer.__class__.__name__}"
                    layer.register_forward_hook(save_outputs_hook(layer_id))
        
        register_hooks(online_model.backbone, 'backbone')
        register_hooks(online_model.neck, 'neck')
        register_hooks(online_model.head, 'head')
        
        # forward
        batch = self.buffer.sample(self.cfg.batch_size, self.n_step, self.gamma, self.prior_weight)
        batch['obs'].requires_grad=True
        rl_loss, preds, targets = self.forward(online_model, target_model, batch, mode='eval')
        
        # explained variance    
        pred_var = torch.var(preds)
        target_var = torch.var(targets)
        exp_var = explained_variance(preds, targets)

        ##########################
        # Smoothness
        # smoothness of prediction
        # grad_norm: w.r.t the input
        grads = torch.autograd.grad(
            outputs=preds, inputs=batch['obs'], grad_outputs=torch.ones_like(preds), 
            create_graph=False, retain_graph=True, allow_unused=True
        )[0]
        pred_grad_norm = torch.mean(grads.flatten(1).norm(2, -1))

        # fisher trace: w.r.t the parameter
        pred_fisher_trace = 0
        params = [param for param in self.model.parameters() if param.requires_grad]
        for param in params:
            grads = torch.autograd.grad(
                outputs=preds, inputs=param, grad_outputs=torch.ones_like(preds),  
                create_graph=False, retain_graph=True, allow_unused=True
            )[0]            
            if grads is not None:
                pred_fisher_trace += (grads ** 2).mean()

        # smoothness of loss
        # grad_norm: w.r.t the input
        grads = torch.autograd.grad(
            outputs=rl_loss, inputs=batch['obs'], 
            create_graph=False, retain_graph=True, allow_unused=True
        )[0]
        loss_grad_norm = torch.mean(grads.flatten(1).norm(2, -1))

        # fisher trace: w.r.t the parameter
        loss_fisher_trace = 0
        params = [param for param in self.model.parameters() if param.requires_grad]
        for param in params:
            grads = torch.autograd.grad(
                outputs=rl_loss, inputs=param,
                create_graph=False, retain_graph=True, allow_unused=True
            )[0]            
            if grads is not None:
                loss_fisher_trace += (grads ** 2).mean()

        # parameter distance
        param_dist, backbone_param_dist, neck_param_dist, head_param_dist = 0.0, 0.0, 0.0, 0.0
        for (key, initial), (_, online) in zip(
            self.initial_model.named_parameters(), 
            online_model.named_parameters()
            ):
            dist = ((initial - online)**2).sum()
            if 'backbone' in key:
                backbone_param_dist += dist
            elif 'neck' in key:
                neck_param_dist += dist
            elif 'head' in key:
                head_param_dist += dist
                
            param_dist += dist

        param_dist = torch.sqrt(param_dist)
        backbone_param_dist = torch.sqrt(backbone_param_dist)
        neck_param_dist = torch.sqrt(neck_param_dist)
        head_param_dist = torch.sqrt(head_param_dist)
            
        # weight_norm
        weight_norm, backbone_weight_norm, neck_weight_norm, head_weight_norm= 0.0, 0.0, 0.0, 0.0
        for key, param in online_model.named_parameters():
            norm = ((param)**2).sum()
            if 'backbone' in key:
                backbone_weight_norm += norm
            elif 'neck' in key:
                neck_weight_norm += norm
            elif 'head' in key:
                head_weight_norm += norm
            weight_norm += norm
            
        weight_norm = torch.sqrt(weight_norm)
        backbone_weight_norm = torch.sqrt(backbone_weight_norm)
        neck_weight_norm = torch.sqrt(neck_weight_norm)
        head_weight_norm = torch.sqrt(head_weight_norm)
        
        #############################
        # log evaluated metrics
        eval_logs = {
            'pred_var': pred_var.item(),
            'target_var': target_var.item(),
            'exp_var': exp_var.item(),
            'rl_loss': rl_loss.item(),
            'pred_grad_norm': pred_grad_norm.item(),
            'pred_fisher_trace': pred_fisher_trace.item(),      
            'loss_grad_norm': loss_grad_norm.item(),
            'loss_fisher_trace': loss_fisher_trace.item(),
            'param_dist': param_dist.item(),
            'backbone_param_dist': backbone_param_dist.item(),
            'neck_param_dist': neck_param_dist.item(),
            'head_param_dist': head_param_dist.item(),
            'weight_norm': weight_norm.item(),
            'backbone_weight_norm': backbone_weight_norm.item(),
            'neck_weight_norm': neck_weight_norm.item(),
            'head_weight_norm': head_weight_norm.item(),
        }
        
        if self.cfg.plot_weight_histogram:
            weight_histogram= {}
            for layer_name, param in online_model.named_parameters():
                hist, edges = np.histogram(param.flatten().abs().cpu().detach().numpy(), bins = 50)
                histogram = visualize_histogram(hist, edges)
                weight_histogram['weight_' + layer_name] = wandb.Image(histogram)
            for layer_name, activation in layer_wise_outputs.items(): 
                hist, edges = np.histogram(activation[0].flatten().cpu().detach().numpy(), bins = 50)
                histogram = visualize_histogram(hist, edges)
                weight_histogram['activation_' + layer_name] = wandb.Image(histogram)
            
            eval_logs.update(weight_histogram)

        return eval_logs 
        

    def rollout(self):
        if self.cfg.rollout_model == 'online':
            rollout_model = self.model
        else:
            rollout_model = self.target_model   
            
        game_id = self.eval_env.game_id # (n, )
        game_id = torch.LongTensor(game_id).to(self.device)
        game_id = rearrange(game_id, 'n -> n 1')       
        
        obs = self.eval_env.reset() # (n, f, c, h, w)
        frames = deque([], maxlen=1000)
        while True:     
            # log first observation
            # obs: (c,h,w)
            frames.append(obs[0][-1])
            
            # encode last observation to torch.tensor()
            obs_tensor = self.buffer.encode_obs(obs)

            # evaluation is based on greedy prediction
            with torch.no_grad():
                action = self.predict(rollout_model, obs_tensor, self.cfg.eval_eps)

            # step
            next_obs, reward, done, info = self.eval_env.step(action.reshape(-1))

            # logger
            self.logger.step(obs, reward, done, info, mode='eval')
            
            # move on
            if self.logger.is_traj_done(mode='eval'):
                break
            else:
                obs = next_obs

        video = np.array(frames)
        if video.shape[1] == 1:
            video = video.repeat(3,1)
            
        #rollout_logs = {
        #    'video': wandb.Video(video, fps=12, format='gif')
        #}
        rollout_logs = {}
        
        return rollout_logs
        
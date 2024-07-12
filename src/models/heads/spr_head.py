from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead
from einops import rearrange, repeat

class SPRHead(BaseHead):
    name = 'spr_head'
    def __init__(self,
                 in_shape,
                 action_size,
                 num_heads,
                 num_predictions,
                 num_actions):

        super().__init__(in_shape, action_size)
        self.in_dim = in_shape[0]
        self.num_heads = num_heads
        self.num_predictions = num_predictions
        self.num_actions = num_actions

        # Transition model
        self.action_embed = nn.Embedding(num_heads*num_actions, num_actions) # Gamewise action embedding
        self.transition = nn.RNN(input_size=num_actions,
                                 hidden_size=self.in_dim,
                                 num_layers=1,
                                 bias=True,
                                 batch_first=True)

        # Gamewise predictor model
        self.weights = nn.Embedding(num_heads, self.in_dim * action_size)
        self.biases = nn.Embedding(num_heads, action_size)

    def forward(self, x, act=None, idx=None):
        """
        [params] x (torch.Tensor: (n, 1, d))
        [params] act (torch.Tensor: (n, t-1)) performed actions in this t steps (t-1 = num_predictions)
        [params] idx (torch.Tensor: (n, t-1)) idx of each head to utilize
        [returns] x (torch.Tensor: (n, t-1, a))
        """

        n, _, d = x.shape
        t = self.num_predictions
        a = self.action_size
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
        if act is None:
            act = torch.zeros((n, t), device=x.device).long()
        
        # Game-wise action embedding
        act_id = act + self.num_actions*idx
        action_embed = self.action_embed(act_id)
            
        x = rearrange(x, 'n 1 d -> 1 n d')

        # Transition
        x_t, _ = self.transition(action_embed, x)

        # Gamewise linear
        weights = self.weights(idx)
        biases = self.biases(idx)
        weights = rearrange(weights, 'n t (d a) -> (n t) d a', d=d, a=a)
        biases = rearrange(biases, 'n t a -> (n t) a')
        x_t = rearrange(x_t, 'n t d -> (n t) 1 d')
        
        x_t = torch.bmm(x_t, weights).squeeze(1) + biases
        x_t = rearrange(x_t, '(n t) a -> n t a', t=t)   
 
        info = {}
        return x_t, info
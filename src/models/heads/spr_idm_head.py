from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead
from einops import rearrange, repeat

class SPRIDMHead(BaseHead):
    name = 'spr_idm_head'

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

        # Gamewise predictor model (SPR)
        self.spr_w = nn.Embedding(num_heads, self.in_dim * action_size) # action_size := embed_dim
        self.spr_b = nn.Embedding(num_heads, action_size)

        # Gamewise predictor model (IDM)
        self.idm_w1 = nn.Embedding(num_heads, 2*self.in_dim * action_size)
        self.idm_b1 = nn.Embedding(num_heads, action_size)
        self.idm_w2 = nn.Embedding(num_heads, action_size * num_actions) # num_actions := actual action dim
        self.idm_b2 = nn.Embedding(num_heads, num_actions)

    def forward(self, x, x_target=None, act=None, idx=None):
        """
        [params] x (torch.Tensor: (n, 1, d))
        [params] x_target (torch.Tensor: (n, t-1, d)) input for IDM prediction
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
        if x_target is None:
            x_target = torch.zeros()
        
        # Game-wise action embedding
        act_id = act + self.num_actions*idx
        action_embed = self.action_embed(act_id)
    
        # Transition
        transition_inp = rearrange(x, 'n 1 d -> 1 n d')
        x_t, _ = self.transition(action_embed, transition_inp)
        
        # SPR prediction
        spr_w = rearrange(self.spr_w(idx), 'n t (d a) -> (n t) d a', d=d, a=a)
        spr_b = rearrange(self.spr_b(idx), 'n t a -> (n t) a')
        spr_x = rearrange(x_t, 'n t d -> (n t) 1 d')

        spr_pred = torch.bmm(spr_x, spr_w).squeeze(1) + spr_b
        spr_pred = rearrange(spr_pred, '(n t) a -> n t a', t=t)

        # IDM prediction
        idm_x = torch.cat((x, x_t[:,:-1,:]), dim=1)
        idm_inp = torch.cat((idm_x, x_target), dim=2) # [n t-1 2*d]
        idm_w1 = rearrange(self.idm_w1(idx), 'n t (d2 a) -> (n t) d2 a', d2=d*2, a=a)
        idm_b1 = rearrange(self.idm_b1(idx), 'n t a -> (n t) a')
        idm_w2 = rearrange(self.idm_w2(idx), 'n t (a o) -> (n t) a o', a=a, o=self.num_actions)
        idm_b2 = rearrange(self.idm_b2(idx), 'n t o -> (n t) o')

        idm_inp = rearrange(idm_inp, 'n t d2 -> (n t) 1 d2')
        idm_pred = torch.bmm(idm_inp, idm_w1).squeeze(1) + idm_b1
        idm_pred = F.relu(idm_pred, inplace=True)
        idm_pred = rearrange(idm_pred, 'nt a -> nt 1 a')
        idm_pred = torch.bmm(idm_pred, idm_w2).squeeze(1) + idm_b2
        idm_pred = rearrange(idm_pred, '(n t) o -> n t o', t=t)

        info = {'idm_pred': idm_pred}

        return spr_pred, info
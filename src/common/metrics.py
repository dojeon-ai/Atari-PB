import torch

EPS = 1e-7

def explained_variance(x, y):
    """
    (params) x: torch.Tensor (,)
    (params) y: torch.Tensor (,)
    """
    x_var = torch.var(x)
    y_var = torch.var(y)
    diff_var = torch.var(x - y)
    exp_var = 1 - diff_var / (y_var + EPS)
    
    return exp_var


###########################################
# Dormant Neurons
# https://arxiv.org/pdf/2302.12902.pdf
# measure the dormancy with smooth rank measure from Roy & Vetterli
def dormant_neurons(activation):
    """
    (params) activation: torch.Tensor (n, d)
    """
    n, d = activation.shape
    S = activation.abs().mean(0) # (d,)
    P = S / S.sum() + EPS
    rank = (-P@(P.log())).exp()
    dormant_rates = (d - rank) / d

    return dormant_rates


#########################################
# RankMe
# https://openreview.net/forum?id=uGEBxC8dnEh
def rankme(activation):
    """
    (params) activation: torch.Tensor (n, d)
    """    
    S = torch.linalg.svdvals(activation)
    P = S / S.sum() + EPS
    rank = (-P@(P.log())).exp()

    return rank, S
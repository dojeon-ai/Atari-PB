from .base import BaseHead
from .identity import IdentityHead
from .linear import LinearHead
from .mh_linear import MHLinearHead
from .mh_distributional import MHDistributionalHead
from .mae_head import MAEHead
from .siammae_head import SiamMAEHead
from .spr_head import SPRHead
from .spr_idm_head import SPRIDMHead

__all__ = [
    'BaseHead',
    'IdentityHead',
    'LinearHead',
    'MHLinearHead',
    'MHDistributionalHead',
    'MAEHead',
    'SiamMAEHead',
    'SPRHead',
    'SPRIDMHead'
]

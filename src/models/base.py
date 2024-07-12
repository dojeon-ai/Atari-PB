import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def encode(self, x):
        b, b_info = self.backbone(x)
        n, n_info = self.neck(b)
        
        info = {
            'backbone': b_info,
            'neck': n_info,
        }

        return n, info        

    def forward(self, x):
        """
        [backbone]: (n,t,f,c,h,w)-> Tuple((n,t,c,h,w), info)
        [neck]: (n,t,c,h,w)-> Tuple((n,t,d), info)
        [head]: (n,t,d)-> Tuple((n,t,d), info)
        """
        b, b_info = self.backbone(x)
        n, n_info = self.neck(b)
        h, h_info = self.head(n)
        
        info = {
            'backbone': b_info,
            'neck': n_info,
            'head': h_info,
        }
        
        return h, info
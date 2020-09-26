import torch
import torch.nn as nn
from norm import Norm
from layer import DisBlock
import numpy as np
from activation import Activation

class Discriminator(nn.Module):
    
    def __init__(self, img_size : int, norm : str, act : str,
                 spectral : bool = True, noise : bool = True):
                 
        super(Discriminator, self).__init__()
        
        i = int(img_size * 8)
        self.num = []
        while len(self.num) < (np.log2(img_size) - np.log2(4) - 1):
            self.num.append(int(i // 2))
            i /= 2.
        self.num.reverse()
        
        
        self.blocks = [DisBlock(x, norm, act, spectral, noise) for x in self.num]
        
        self.net = nn.Sequential(*[nn.Conv2d(3, self.num[0], 4, 2, 1, bias = False),
                                   Norm(norm, self.num[0]),
                                   nn.LeakyReLU(),
                                   *self.blocks,
                                   nn.Conv2d(self.num[-1] * 2, 1, 4, 1, 0, bias=False),
                                   nn.Sigmoid()
                                   ])
        
        
    def forward(self, x : torch.Tensor):
        return(self.net(x))
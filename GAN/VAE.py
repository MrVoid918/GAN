import torch
import torch.nn as nn
from norm import Norm
from layer import DisBlock
import numpy as np

class Encoder(nn.Module):
    
    def __init__(self, img_size : int, norm : str,):
                 #spectral : bool = True, noise : bool = True):
                 
        super(Encoder, self).__init__()
        
        i = int(img_size * 8)
        self.num = []
        while len(self.num) < (np.log2(img_size) - np.log2(4) - 1):
            self.num.append(int(i // 2))
            i /= 2.
        self.num.reverse()
        
        
        self.blocks = [DisBlock(x, norm, 0, 0) for x in self.num]
        
        self.net = nn.Sequential(*[nn.Conv2d(3, self.num[0], 4, 2, 1, bias = False),
                                   Norm(norm, self.num[0]),
                                   nn.LeakyReLU(),
                                   *self.blocks
                                   ])
        
        self.flat = nn.Flatten()
        self.lin = self.flat(self.net(torch.randn(1, 3, img_size, img_size))).shape[-1]
        self.mu = nn.Linear(self.lin, 128)
        self.logvar = nn.Linear(self.lin, 128)
        
        
    def forward(self, x : torch.Tensor):
        x = self.net(x)
        x = self.flat(x)
        mu, logvar = self.mu(x), self.logvar(x)
        
        return(mu, logvar)
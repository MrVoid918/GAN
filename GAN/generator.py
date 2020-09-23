import torch.nn as nn
from layer import Upsample_Block, ResBlock
from norm import Norm
from functools import partial
import torch
import numpy as np

class Generator(nn.Module):
    
    blocks = {'up' : partial(Upsample_Block),
              'res' : partial(ResBlock, upsample = True)}
    
    def __init__(self, img_size : int, norm : str, 
                 up_type : str = 'up', device : str = 'cpu'):
        
        super(Generator, self).__init__()
        
        i = int(img_size * 8)
        self.num = []
        while len(self.num) < (np.log2(img_size) - np.log2(4) - 1):
            self.num.append(int(i))
            i /= 2.
            
        self.blocks_ = [Generator.blocks[up_type](i, norm = norm).to(device) for i in self.num]
        self.blocks_ = nn.ModuleList(self.blocks_)
        
        self.conv1 = nn.ConvTranspose2d(128, self.num[0], 4, 1, 0, bias=False)
        self.norm1 = Norm(norm, self.num[0])
        self.relu = nn.ReLU()
        self.conv_last = nn.ConvTranspose2d(self.num[-1] // 2, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, latent : torch.Tensor):
        x = self.conv1(latent)
        x = self.norm1(x, latent)
        x = self.relu(x)
        for block in self.blocks_:
            x = block(x, latent)
        x = self.conv_last(x)
        return self.tanh(x)
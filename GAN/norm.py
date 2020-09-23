import torch.nn as nn
from inspect import signature
import torch
import torch.nn.functional as F
from functools import partial

            
class SelfModulationBN2D(nn.Module):
    
    def __init__(self, in_features : int, latent_size : int = 128):
        super().__init__()
        
        self.WMLP = nn.Linear(latent_size, 1)
        self.BMLP = nn.Linear(latent_size, 1)
        self.in_features = in_features
        self.norm = nn.BatchNorm2d(in_features, affine = False)
                
    def forward(self, x : torch.Tensor, latent : torch.Tensor):
        minibatch = latent.shape[0]
        size = x.shape[-1]
        weight = self.WMLP(latent.reshape(minibatch, -1))
        weight = F.leaky_relu(weight).expand(minibatch, self.in_features)[:, :, None, None]
        bias = self.BMLP(latent.reshape(minibatch, -1))
        bias = F.leaky_relu(bias).expand(minibatch, self.in_features)[:, :, None, None]
        x = weight * self.norm(x) + bias
        return(x) 

class Norm(nn.Module):
    
    norms = {'BN': partial(nn.BatchNorm2d),
             'IN': partial(nn.InstanceNorm2d),
             #'SN': partial(SwitchNorm2d),
             'SMBN' : partial(SelfModulationBN2D),
             'None': nn.Identity}
    
    def __init__(self, norm : str, in_features : int, latent_size : int = 128):
        super(Norm, self).__init__()
        if norm not in Norm.norms.keys():
            raise NotImplementedError('normalization layer is not found')
        
        self.norm = norm
        self.in_features = in_features
        self.norm_layer = Norm.norms[norm](in_features)
        
    def forward(self, x : torch.Tensor, latent : torch.Tensor = None):
        n_params = signature(self.norm_layer.forward).parameters
        if len(n_params) == 2:
            return self.norm_layer(x, latent)   #Ugly bypass to check if norm is dependent on latent
        else:
            return self.norm_layer(x)

import torch.nn as nn
import torch
from norm import Norm
from functools import partial
from collections import OrderedDict
from torch.nn.utils import spectral_norm
from activation import Activation

class Upsample_Block(nn.Module):
    
    spectral_layer = {True : partial(spectral_norm),
                      False : nn.Identity()}
    
    def __init__(self, 
                 in_features : int,
                 norm : str,
                 act : str,
                 spectral : bool
                 ):
        
        super(Upsample_Block, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.reflection = nn.ReflectionPad2d(1)
        self.spectral = Upsample_Block.spectral_layer[spectral]
        
        self.conv = self.spectral(nn.Conv2d(in_features, in_features // 2, kernel_size = 3, 
                                  stride = 1, padding = 0, bias = False))
        self.norm = Norm(norm, in_features // 2)
        self.act = Activation(act)
        
    def forward(self, x : torch.Tensor, latent : torch.Tensor):
        x = self.upsample(x)
        x = self.reflection(x)
        x = self.conv(x)
        x = self.norm(x, latent)    
        x = self.act(x)
        
        return x
        
class Deconv_Block(nn.Module):
    
    spectral_layer = {True : partial(spectral_norm),
                      False : nn.Identity()}    
    
    def __init__(self,
                 in_features : int,
                 norm : str,
                 act : str,
                 spectral : bool):
        
        super(Deconv_Block, self).__init__()
        
        self.spectral = Deconv_Block.spectral_layer[spectral]
        
        self.deconv = self.spectral(
                      nn.ConvTranspose2d(in_features, in_features // 2, 4, 2, 1, bias=False))
        self.norm = Norm(norm, in_features // 2)
        self.act = Activation(act)
        
    def forward(self, x : torch.Tensor, latent : torch.Tensor):
        x = self.deconv(x)
        x = self.norm(x, latent)
        x = self.act(x)
        
        return x
        
class ResBlock(nn.Module):
    
    spectral_layer = {True : partial(spectral_norm),
                      False : nn.Identity()}  
    
    def __init__(self,
                 in_channels : int, 
                 norm : str,
                 act : str,
                 spectral : bool,
                 upsample : bool,         
                 hidden_channels=None,):
                 
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.spectral = ResBlock.spectral_layer[spectral]
        self.upsample = upsample
        self.out_channels = in_channels // 2
        self.conv1 = self.spectral(
                     nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        self.conv2 = self.spectral(
                     nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1))
        self.conv_sc = self.spectral(
                     nn.Conv2d(in_channels, self.out_channels, kernel_size=1, padding=0))
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.norm1 = Norm(norm, in_channels)
        self.norm2 = Norm(norm, hidden_channels)
        self.act = Activation(act)
        
    def forward_residual_connect(self, input):
        out = self.conv_sc(input)
        if self.upsample:
             out = self.upsampling(out)
            #out = self.upconv2(out)
        return out
    
    def forward(self, input : torch.Tensor, latent : torch.Tensor):
        out = self.act(self.norm1(input, latent))
        out = self.conv1(out)
        if self.upsample:
             out = self.upsampling(out)
             #out = self.upconv1(out)
        out = self.act(self.norm2(out, latent))
        out = self.conv2(out)
        out_res = self.forward_residual_connect(input)
        return out + out_res
        
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 
        

class DisBlock(nn.Module):
    
    spectral_layer = {True : partial(spectral_norm),
                      False : nn.Identity()}
                
    noise_layer = {True : GaussianNoise(),
                   False : nn.Identity()}
    
    def __init__(self,
                 in_features : int ,
                 norm : str,
                 act : str,
                 spectral : bool = True,
                 noise : bool = True,
                 dropout : float = 0.):
        
        super(DisBlock, self).__init__()
        self.spectral = DisBlock.spectral_layer[spectral]
        
        self.dis_block = nn.Sequential(OrderedDict([
        ('conv', self.spectral(nn.Conv2d(in_features, in_features * 2, 4, 2, 1, bias = False))),
        ('noise', DisBlock.noise_layer[noise]),
        ('norm', Norm(norm, in_features * 2)),
        ('act', Activation(act)),
        ('dropout', nn.Dropout2d(dropout))]
        ))
        
        
    def forward(self, x : torch.Tensor):
        return (self.dis_block(x))
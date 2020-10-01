from collections import OrderedDict
from functools import partial
from self_modulation import SelfModulationBN2D
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from norm import Norm

norms = {'BN': partial(nn.BatchNorm2d),
         'IN': partial(nn.InstanceNorm2d),
         #'SN': partial(SwitchNorm2d),
         'SMBN' : partial(SelfModulationBN2D),
         'None': nn.Identity}

class deconv_block(nn.Module):
    
    def __init__(self, in_f, activation = nn.ReLU(inplace = True), norm = 'BN'):
        super().__init__()
        
        
        if norm not in norms.keys():
            raise NotImplementedError('normalization layer is not found')
            
        self.norm = norm

        self.block = nn.Sequential(OrderedDict([
        ("deconv", nn.ConvTranspose2d(in_f ,in_f // 2, 4, 2, 1, bias=False)),
        ("norm_", norms[self.norm](in_f // 2)),
        ("non_linear", activation)
            ]
          )
        )
        
    def forward(self, x):
        return self.block(x)
    
class upsample_block(nn.Module):
    
    def __init__(self, 
                 in_f : int,
                 activation = nn.ReLU(inplace = True),
                 norm = 'BN',
                 spectral : bool):
                 
        super().__init__()
        
        if norm not in norms.keys():
            raise NotImplementedError('normalization layer is not found')
            
        if spectral:
            self.spec = partial(spectral_norm)
        else:
            self.spec = partial(nn.Identity)
       
        self.norm_ = norm
        self.block = nn.Sequential(OrderedDict([
            ("upsample", nn.Upsample(scale_factor = 2)),
            ("reflection", nn.ReflectionPad2d(1)),
            ("conv", self.spec(nn.Conv2d(in_f, in_f // 2, kernel_size = 3, stride = 1, padding = 0, bias = False))])))
        self.norm = norms[norm](in_f // 2)
        self.non_linear = activation
        
    def forward(self, x, latent):
        x = self.block(x)
        if 'SM' in self.norm_:
            x = self.norm(x, latent)
        else:
            x = self.norm(x)
            
        x = self.non_linear(x)
        return (x)
       
class SMGenerator(nn.Module):
    
    def __init__(self, img_size, activation = nn.LeakyReLU(0.2, inplace = True), 
                 gen_method = "deconv", norm = 'SMBN'):
        super().__init__()

        i = img_size * 8
        self.num = []
        while len(self.num) < (np.log2(ngf) - np.log2(4) - 1):
            self.num.append(int(i))
            i /= 2.
        
        self.gen_method = gen_method
                
        if self.gen_method == "upsample":
            self.gen_blocks = nn.ModuleList([upsample_block(x, activation = activation, norm = norm) 
                                            for x in self.num])
            
        self.conv1 = nn.ConvTranspose2d(nz, img_size * 8, 4, 1, 0, bias=False)
        self.norm1 = norms[norm](img_size * 8)
        self.act1 = activation
        self.conv_last = nn.ConvTranspose2d(self.num[-1] // 2, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x, latent):
        x = self.conv1(x)
        x = self.norm1(x, latent)
        x = self.act1(x)
        for blocks in self.gen_blocks:
            x = blocks(x, latent)
        x = self.conv_last(x)
        return self.tanh(x)
        
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer = 8, spectral = False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = spectral_norm(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, 
                                                 stride = 2, padding=1, groups=nin, bias = False))
        self.pointwise = spectral_norm(nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, bias = False))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class dis_block(nn.Module):
    def __init__(self, in_c, activation = nn.LeakyReLU(0.2, inplace = True), 
                 depth_sep = False, spectral = False, kernels_per_layer = 8):
        super().__init__()
        
        if spectral:
            self.spec = partial(spectral_norm)
        else:
            self.spec = partial(nn.Identity)
        
        if depth_sep:
            self.Dconv = depthwise_separable_conv(nin = in_c, kernels_per_layer = kernels_per_layer, 
                                                  nout = in_c * 2)
        else:
            self.Dconv = self.spec(nn.Conv2d(in_c, in_c * 2, 4, 2, 1, bias = False))

        self.activation = activation
        
        self.dis_block = nn.Sequential(OrderedDict([
            #spectral_norm(nn.Conv2d(in_c, in_c * 2, 4, 2, 1, bias = False)),
            #nn.Conv2d(in_c, in_c * 2, 4, 2, 1, bias = False),
            ("noise", GaussianNoise()),
            ("conv_block", self.Dconv),
            ("norm", nn.BatchNorm2d(in_c * 2)),
            ("non_linear", self.activation),
            ("dropout", nn.Dropout2d(0.25))
            ]
          )
        )
        
    def forward(self, x):
        return self.dis_block(x)

class Encoder_Discriminator(nn.Module):
    
    def __init__(self, img_size: int, 
                activation = nn.LeakyReLU(0.2, inplace = True),
                depth_sep = False,    
                loss = "adv",
                spectral = True):
                
        super().__init__()
        
        self.depth_sep = depth_sep
        
        if spectral:
            self.spec = partial(spectral_norm)
        else:
            self.spec = partial(nn.Identity)

        self.blocks = [dis_block(x, activation = activation, depth_sep = depth_sep, spectral = spectral)
                       for x in np.rint(np.geomspace(img_size // 4, img_size * 2, 4)).astype(int)]
                           

        self.sequence = [self.spec(nn.Conv2d(nc, img_size // 4, 4, 2, 1, bias = False)),
                          nn.BatchNorm2d(img_size // 4),
                          activation,
                          nn.Dropout2d(0.25),
                          GaussianNoise(),
                          *self.blocks]
                          #self.spec(nn.Conv2d(img_size * 4, 1, 4, 1, 0, bias=False)),
                          #nn.Sigmoid()]
                          
        ###Output is no longer scalar, rather we pass as feature vector
        
        self.net = nn.Sequential(*self.sequence)
        
    def forward(self, x):
        return self.net(x)
        
class Encoder(nn.Module):
    
    def __init_(self, img_size: int, n_latent = 128):
        super().__init__()
        
        self.sequence = [spectral_norm(nn.Conv2d(nc, img_size // 4, 4, 2, 1, bias = False)),
                          nn.BatchNorm2d(img_size // 4),
                          activation,
                          nn.Dropout2d(0.25),
                          GaussianNoise(),
                          *self.blocks,
                          ]
        
        self.net = nn.Sequential(*self.sequence)
        
        self.fc_mu = nn.Linear(img_size * 4, n_latent)
        self.fc_var = nn.Linear(img_size * 4, n_latent)
        
    def encode(self, input, mu_var = True):
        result = self.encoder(input)
        if mu_var:
            mu = self.fc_mu(result)
            var = self.fc_var(result)
            return [mu, var]
        else:
            return(result)

class Enc_Dis_Linear(nn.Module):
    def __init__(self, img_size = 128):
        
        super().__init__()
        
        self.net = nn.Linear(img_size * 4 * 2, 1)
        
    def forward(self, x):
        return(F.sigmoid(self.net(x)))
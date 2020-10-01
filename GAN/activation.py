import torch.nn as nn

class Activation(nn.Module):
    
    activations = nn.ModuleDict([['relu', nn.ReLU(inplace = True)],
                                 ['lrelu', nn.LeakyReLU(0.2, inplace = True)],
                                 ['selu', nn.SELU(inplace = True)]])
                                 
    def __init__(self, act : str):
        
        super(Activation, self).__init__()
        
        if act not in Activation.activations.keys():
            raise NotImplementedError('activation layer is not found')
        
        self.activation = Activation.activations[act]
        
    def forward(self, x):
        return self.activation(x)
    
    def __repr__(self):
        return self.activation.__class__.__name__
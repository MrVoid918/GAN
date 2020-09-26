import torch.nn as nn

class Activation(nn.Module):
    
    activations = nn.ModuleDict([['relu', nn.ReLU()],
                                 ['lrelu', nn.LeakyReLU()],
                                 ['selu', nn.SELU()]])
                                 
    def __init__(self, act : str):
        
        super(Activation, self).__init__()
        
        if act not in Activation.activations.keys():
            raise NotImplementedError('activation layer is not found')
        
        self.activation = Activation.activations[act]
        
    def forward(self, x):
        return self.activation(x)
import torch.nn.functional as F
import torch
from functools import partial
from pytorch_msssim import ms_ssim


class Loss:
    
    def LSLoss(output : torch.Tensor, label :torch.Tensor):
        return 0.5 * torch.mean((output - label)**2)
    
    def AdvLoss(output : torch.Tensor = None, label :torch.Tensor = None):
        return partial(F.binary_cross_entropy)
    
    def RelAdvLoss(output : torch.Tensor = None, label :torch.Tensor = None):
        return partial(F.binary_cross_entropy_with_logits)
            
    
    losses = {'adv': AdvLoss(),
              'rel': RelAdvLoss(),
              'ls': partial(LSLoss)}
    
    def __init__(self, loss_type:str):
        
        if loss_type not in Loss.losses.keys():
            raise NotImplementedError("Loss function isn't defined")
        
        self.loss_type = loss_type
        self.loss = Loss.losses[self.loss_type]
        
    def __call__(self, output, label):
        return self.loss(output, label)


def VAELoss(recons : torch.Tensor,
            _input : torch.Tensor,
            mu : torch.Tensor,
            log_var : torch.Tensor,
            kld_weight : float = 1.,
            loss : str = 'MSE'):
    
    losses = {'MSE': partial(F.mse_loss),
              'logcosh' : partial(_logcosh),
              'MS-SSIM': partial(_ms_ssim)}
    
    if loss not in losses.keys():
        raise NotImplementedError("Loss function isn't defined")
            
    recons_loss = losses[loss](recons, _input)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return recons_loss, kld_loss

def _ms_ssim(recon, _input):
    recon = (recon + 1) / 2
    _input = (_input + 1) / 2
    return ms_ssim(recon, _input, data_range = 1, size_average = 1)
    
def _logcosh(recon, _input, alpha = 100):
    t = recon - _input
    recons_loss = alpha * t + \
                  torch.log(1. + torch.exp(- 2 * alpha * t)) - \
                  torch.log(torch.tensor(2.0))
    recons_loss = (1. / alpha) * recons_loss.mean()
    
    return recons_loss
import torch
import os

def reparamaterize(mu : torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    return eps * std + mu
    
def save_state(save_dir : str, epoch : int, G, D):
  G_path = os.path.join(save_dir, "{}_G.pth".format(epoch))
  D_path = os.path.join(save_dir, "{}_D.pth".format(epoch))
  torch.save(G.state_dict(), G_path)
  torch.save(D.state_dict(), D_path)
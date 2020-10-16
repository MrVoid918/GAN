import torch
import os
import glob
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from IPython import display
from torch.autograd import Variable, grad

def reparamaterize(mu : torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
    
def save_state(save_dir : str, epoch : int, G, D):
    G_path = os.path.join(save_dir, "{}_G.pth".format(epoch))
    D_path = os.path.join(save_dir, "{}_D.pth".format(epoch))
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def save_images(tensor, Generator, img):
    with torch.no_grad():
        fake = Generator(tensor).detach().cpu()
        img.append(vutils.make_grid(fake, padding=2, normalize=True))
        
def show_images(img):
    InteractiveShell.ast_node_interactivity = "all"
    
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)

    writer = animation.PillowWriter(fps = 25)
    ani.save("results.gif", writer = writer)
    
    plt.close(ani._fig)
    with open("results.gif",'rb') as f:
        display.Image(data=f.read(), format='png')
        
def gradient_penalty(image_data, discriminator, device, lambda_ = 10.):
    #image to get dimensions
    #https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
    image = image_data
    image.requires_grad = True
    batch_size = image.shape[0]
    alpha = torch.randn(image.size()).to(device)
    x_hat = alpha * image + (1 - alpha) * (image + 0.5 * image.std() * torch.rand(image.size()).to(device))
    pred_hat = discriminator(x_hat)
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
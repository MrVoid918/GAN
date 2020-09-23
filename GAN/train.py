import torch
import torch.nn as nn
from argparse import ArgumentParser
from utils import save_state
from pathlib import Path
from data import load_data
from init import weights_init
from loss import Loss
import torch.optim as optim
from norm import Norm
from generator import Generator
from discriminator import Discriminator

def parse_args():
    parser = ArgumentParser(prog="GAN")

    parser.add_argument("--img_dir", type=str, required=1, help="File Directory for Dataset")
    parser.add_argument("--img_size", type=int, default=128, help="Image Size")
    parser.add_argument("--batch_size", type = int, default=16, help="Batch Size")
    parser.add_argument("--norm", type = str, default="BN", choices = Norm.norms.keys(), help="Norm Type")
    parser.add_argument("--up_type", type = str, default="up", choices = ["up", "res"], help="Options for generator upsampling. up, res")
    parser.add_argument("--spectral", type = bool, default=1, help="Spectral Norm for Discriminator")
    parser.add_argument("--noise", type=bool, default=0, help="Gaussian Noise for Discriminator")
    parser.add_argument("--loss", type=str, default="adv", choices = Loss.losses.keys(), help="Loss Function")
    parser.add_argument("--device", type=str, default="cpu", choices = ["cpu", "gpu"], help="Device to run on. Defaults on cpu. gpu:0")
    parser.add_argument("--epoch", type=int, default=100, help="Training epochs")
    parser.add_argument("--G_lr", type=float, default=0.002, help="Generator Learning Rate")
    parser.add_argument("--D_lr", type=float, default=0.002, help="Discriminator Learning Rate")
    #arser.add_argument("--save_dir", type=str, required=1, help = "Directory for saved Model")

    return parser.parse_args()


    
     
def main(num_epoch : int, device : str, b_size : int, loss, G, D, optimizerG, optimizerD, data_loader):
    #save_dir : str):   
    print("Starting GAN training\n")   
    for epoch_ in range(num_epoch + 1):
        
        for i, data in enumerate(data_loader, 0):
        
            
            D.zero_grad()

            real_img = data[0].to(device)
            b_size = real_img.size(0)
            label = torch.full((b_size,), real_label, dtype = torch.float32, device=device) #fills size of mini-batch with 1
            output = D(real_img).view(-1)
            loss_D_real = loss(output, label)
            loss_D_real.backward()
            D_x = output.mean().item()
            
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 128, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = D(fake.detach()).view(-1)
            loss_D_fake = loss(output, label)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            loss_D = loss_D_real + loss_D_fake
            # Update D
            optimizerD.step()
            
            ##Train G
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            loss_G = loss(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if i % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch_, num_epoch, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                    
            #if epoch_ % 50 == 0 and epoch_ != 0:
                #save_state(save_dir, epoch_, G, D)
            
if __name__ == "__main__":
    
    args = parse_args()
    
    real_label = 1
    fake_label = 0

    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    data_loader = load_data(args.img_dir, args.img_size, args.batch_size)
    G = Generator(args.img_size, args.norm, args.up_type, device).to(device)
    D = Discriminator(args.img_size, args.norm, args.spectral, args.noise).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    
    optimizerD = optim.Adam(D.parameters(), lr = args.D_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr = args.G_lr, betas=(0.5, 0.999))
    
    loss = Loss(args.loss)
    
    main(args.epoch, device, args.batch_size, loss, G, D, optimizerG, optimizerD, data_loader)
    
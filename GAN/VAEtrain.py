import torch
from utils import reparamaterize
import torch.optim as optim
from loss import VAELoss
from init import weights_init
from generator import Generator
from VAE import Encoder
from activation import Activation

def train_VAE(generator : Generator,
              img_size : int,              
              enc_norm : str,
              act : str,              
              device : str,
              data_loader,
              lr : float,
              num_epoch : int,
              losstype : str = 'MSE'):
              
    dec = generator
    enc = Encoder(img_size, enc_norm, act).to(device)
    
    enc.apply(weights_init)
    
    enc_optimizer = optim.Adam(enc.parameters(), lr = lr, betas=(0.5, 0.999))
    dec_optimizer = optim.Adam(dec.parameters(), lr = lr, betas=(0.5, 0.999))

    print("Starting VAE training\n")
    for epoch in range(num_epoch + 1):
        
        total_recon_loss = 0
        total_kld_loss = 0
    
        for i, data in enumerate(data_loader, 0):
            
            enc.zero_grad()
            dec.zero_grad()
            
            img = data[0].to(device)
            mu, var = enc(img)
            bs = mu.shape[0]
            z = reparamaterize(mu, var).view(bs, -1, 1, 1)
            recon_img = dec(z)
            
            recons_loss, kld_loss = VAELoss(recon_img, img, mu, var, loss = losstype)
            total_recon_loss += recons_loss
            total_kld_loss += kld_loss
            loss = recons_loss + kld_loss
            
            loss.backward(retain_graph = True)
            enc_optimizer.step()
            dec_optimizer.step()
        
                
        print("[Epoch %d/%d] [Reconstruction loss: %f] [KL Divergence loss: %f]"
                % (epoch, num_epoch, total_recon_loss, total_kld_loss)
                )
                
    return enc, dec
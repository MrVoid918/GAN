import torch
from utils import reparamaterize
import torch.optim as optim
from loss import VAELoss
from init import weightS_init

def train_VAE(img_size : int, norm : str, up : str, device : str,
              data_loader, lr : float)
    
    enc = Generator(img_size, norm, up_type, device).to(device)
    dec = Decoder(img_size, norm).to(device)
    
    enc.apply(weights_init)
    dec.apply(weights_init)
    
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
            z = reparamaterize(mu, var)
            recon_img = dec(z)
            
            recons_loss, kld_loss = VAELoss(recon_img, img, mu, var)
            total_recon_loss += recons_loss
            kld_loss += kld_loss
            loss = recons_loss + kld_loss
            
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
        
                
        print("[Epoch %d/%d] [Reconstruction loss: %f] [KL Divergence loss: %f]"
                % (epoch_, num_epoch, total_recon_loss, total_kld_loss)
                )
                
    return enc, dec
from argparse import ArgumentParser
from norm import Norm
from activation import Activation
import argparse
from loss import Loss, VAELoss

class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawTextHelpFormatter
    ):
        pass

def parse_args():
    parser = ArgumentParser(prog="GAN", formatter_class = RawTextArgumentDefaultsHelpFormatter)

    parser.add_argument("--img_dir",
                        type=str,
                        required=1,
                        help="File Directory for Dataset")
                                                    
    parser.add_argument("--save_dir",
                        type=str,
                        required=1,
                        help = "Directory for saved Model")
                        
    parser.add_argument("--img_size",
                        type=int,
                        default=128,
                        help="Image Size")
                        
    parser.add_argument("--batch_size",
                        type = int,
                        default=16,
                        help="Batch Size")
                        
    parser.add_argument("-a", "--act",
                        type = str,
                        default="lrelu",
                        choices = Activation.activations.keys(), help="Activation Type")
                        
    parser.add_argument('-GN', "--G_norm",
                        type = str,
                        default="BN",
                        choices = Norm.norms.keys(),
                        help="Generator Norm")
                        
    parser.add_argument('-DN', "--D_norm",
                        type = str,
                        default="BN",
                        choices = Norm.norms.keys(),
                        help="Discriminator Norm")
                        
    parser.add_argument("-u", "--up_type",
                        type = str,
                        default="up",
                        choices = ["up", "res", "deconv"],
                        help="Options for generator upsampling. up, res")
    
    parser.add_argument("-G_sp", "--Gen_spectral",
                        type = bool,
                        default=1,
                        help="Spectral Norm for Generator")

    parser.add_argument("-D_sp", "--Dis_spectral",
                        type = bool,
                        default=0,
                        help="Spectral Norm for Discriminator")
                        
    parser.add_argument("--noise",
                        type=bool,
                        default=0,
                        help="Gaussian Noise for Discriminator")
                        
    parser.add_argument("-l", "--loss",
                        type=str,
                        default="adv",
                        choices = Loss.losses.keys(),
                        help="Loss Function")
                        
    parser.add_argument("-d", "--device",
                        type=str,
                        default="cpu",
                        choices = ["cpu", "gpu"],
                        help="Device to run on. Defaults on cpu. gpu:0")
                        
    parser.add_argument("-e", "--epoch",
                        type=int,
                        default=100,
                        help="Training epochs")
                        
    parser.add_argument("--G_lr",
                        type=float,
                        default=0.0002,
                        help="Generator Learning Rate")
                        
    parser.add_argument("--D_lr",
                        type=float,
                        default=0.0002,
                        help="Discriminator Learning Rate")
                        
    parser.add_argument("-p", "--pretrain",
                        type=bool,
                        default=0,
                        help="Pretrain with VAE")
                        
    parser.add_argument("--dropout",
                        type=float,
                        default=0.,
                        help="Dropout in Discriminator. Between 0 and 1")
    
    parser.add_argument("-gp", "--gradient_penalty",
                        type=bool,
                        default=0,
                        help="Applies DRAGAN gradient penalty")
                        
    parser.add_argument("-vl", "--vae_loss",
                        type=str,
                        default='MSE',
                        choices = ['MSE', 'logcosh', 'MS-SSIM'],
                        help="Per pixel loss for VAE")
                        
                        

    return parser.parse_args()
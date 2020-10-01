from generator import Generator
from discriminator import Discriminator

class Experiment:

	def __init__(self, G_dir, D_dir, option):
        
        self.options = option
        self.g_dir = G_dir
        self.d_dir = D_dir
        self.G = Generator(option.img_size,
                           option.norm,
                           option.act,
                           option.up, 
        
    def __repr__()
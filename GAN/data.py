from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np


def load_data(dir, img_size : int, batch_size : int):
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
                                    
    dataset = datasets.ImageFolder(root = dir,
                                    transform = transform)
                                    
    data_loader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = True)
                            
    return data_loader
    

class SquarePad:
	def __call__(self, image, value = 255):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, (value, value, value), 'constant')
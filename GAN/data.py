from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
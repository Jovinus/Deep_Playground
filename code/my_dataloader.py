# %%
import torch
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Dataset
# %%

class CustomDataset(Dataset):
    def __init__(self, data, target) -> None:
        super().__init__()
        
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        image = self.data[idx].type(torch.float32).view(1, 28, 28)
        target = self.target[idx].type(torch.LongTensor)
        
        return  image, target

# %%

if __name__ == '__main__':
    
    emnist_train = EMNIST(root="/home/lkh256/Studio/Deep_Playground/data",
                         split='byclass',
                         train=True,
                         download=True)
    
    emnist_valid = EMNIST(root="/home/lkh256/Studio/Deep_Playground/data",
                         split='byclass',
                         train=False,
                         download=True)
    
    train_dataset = CustomDataset(data=emnist_train.data, target=emnist_train.targets)
    valid_dataset = CustomDataset(data=emnist_valid.data, target=emnist_valid.targets)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=4, 
                              shuffle=True, 
                              num_workers=4)
    
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=4, 
                              shuffle=True, 
                              num_workers=4)

# %%

# %% 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from my_dataloader import *

# %%
class CNN_Block(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.cnn_block = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=124, kernel_size=3, padding=1), 
                                       nn.Conv2d(in_channels=124, out_channels=output_channels, kernel_size=3, padding=1),
                                       nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        y = torch.relu(self.cnn_block(x))
        return y

# %%
class CNN_FC_layer(pl.LightningModule):
    def __init__(self, output_class):
        super().__init__()
        
        self.cnn_blocks = nn.Sequential(CNN_Block(input_channels=1, output_channels=100),
                                        nn.ReLU(), 
                                        CNN_Block(input_channels=100, output_channels=200),
                                        nn.ReLU(),
                                        CNN_Block(input_channels=200, output_channels=10),
                                        nn.ReLU(),
                                        nn.Flatten())
        
        self.linear = nn.Sequential(nn.Linear(7840, 1000),
                                    nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(1000, 100),
                                    nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(100, output_class), 
                                    nn.Dropout(p=0.5),)
        
    
    def forward(self, x):
        y_cnn_block = self.cnn_blocks(x)
        y_output = self.linear(y_cnn_block)

        return y_output
        
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
    
    img_train, target_train = next(iter(train_loader))

    cnn_layer = CNN_FC_layer(output_class=62)
    tmp = cnn_layer.forward(img_train)
    print(tmp)
    print(tmp.shape)
# %%

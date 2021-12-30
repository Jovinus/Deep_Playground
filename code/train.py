# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from model import *
# %%
class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('running validation ...')
        return bar

class CNN_Modeling(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.NLLLoss()
        self.softmax = nn.LogSoftmax()
        self.accuracy = Accuracy()
        self.model = CNN_FC_layer(output_class=62)
        
    def forward(self, x):
        logits = self.model(x)
        logits = self.softmax(logits) 
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        J = self.loss(logits, y)
        
        acc =self.accuracy(logits, y)
        
        log = {'train_acc' : acc}
        
        return {'loss':J, 'log':log, 'progress_bar':log}
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        del results['progress_bar']['train_acc']
        return results
    
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
        print(avg_val_acc)
        pbar = {'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_val_loss, 'log':pbar, 'progress_bar': pbar}
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')
# %%
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
                            batch_size=2**10, 
                            shuffle=True, 
                            num_workers=4)

valid_loader = DataLoader(valid_dataset, 
                            batch_size=2**10, 
                            shuffle=True, 
                            num_workers=4)

img_train, target_train = next(iter(train_loader))
# %%

bar = LitProgressBar()

model = CNN_Modeling()

logger = TensorBoardLogger("tb_logs", name="my_model")

trainer = pl.Trainer(logger=logger,
                     progress_bar_refresh_rate = 1, 
                     max_epochs=100, 
                     gpus=1, 
                     gradient_clip_val=0.5, 
                     log_every_n_steps=2, 
                     accumulate_grad_batches=2,
                     callbacks=[bar])
# %%
trainer.fit(model, 
            train_dataloaders = train_loader, 
            val_dataloaders = valid_loader)
# %%

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
import numpy as np
from lib.datamodule import PressurePoseDataModule
from lib.layers import DoubleConv, Down, Up, OutConv, Flatten
from lib.F import mpjpe, pcp, pck
from lib.transforms import HeatmapsToSkeleton, HistogramEqualize, StdMeanNormalize, Stack, Boolean, MinMaxNormalize

class Net(pl.LightningModule):
    def __init__(self, learning_rate=0.00001, unet_checkpoint="tb_logs/unet/version_0/checkpoints/best.ckpt"):
        super().__init__()
        import importlib  
        UNet = importlib.import_module("unet").Net
        SimpleCNN = importlib.import_module("simple-cnn").Net

        self.learning_rate = learning_rate
        
        self.unet = UNet.load_from_checkpoint(unet_checkpoint)

        self.coordinates_extractor = SimpleCNN(in_channels=15)

    def forward(self, im):
        heatmaps = self.unet(im[:,0:3])
        x = torch.cat((heatmaps, im[:,3:4], im[:,1:2]), 1)
        return self.coordinates_extractor(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        loss = mpjpe(y_hat, y)
            
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)
        loss = mpjpe(y_hat, y)
        self.log("val_loss", loss)

        self.log("mpjpe", loss)
        self.log("pcp", pcp(y_hat, y))
        self.log("pck", pck(y_hat, y))

        self.log("lr", self.optimizer.param_groups[0]['lr'])
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch", "monitor": "mpjpe"}]
    
    def on_train_start(self):
        self.train_start_time = time.time()
    
    def on_train_end(self):
        self.train_start_time = None
    
    def on_train_epoch_start(self):
        self.train_epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        self.log("train_epoch_time", time.time() - self.train_epoch_start_time)
        self.log("train_time", time.time() - self.train_start_time)
        self.train_epoch_start_time = None

if __name__ == '__main__':
    dm = PressurePoseDataModule(
        input_transform = Stack([StdMeanNormalize(), Boolean(), HistogramEqualize(), MinMaxNormalize()])
    )

    net = Net()
    net.unet.freeze()
    # net = Net.load_from_checkpoint("tb_logs/combined-net/version_0/checkpoints/best.ckpt")

    dm.setup()
    net.example_input_array = next(iter(dm.val_dataloader()))[0]
    print(ModelSummary(net, max_depth=-1))

    logger = TensorBoardLogger('tb_logs', name='combined-net', log_graph=True)

    best_checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="best")
    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, min_delta=0.02),
            ModelCheckpoint(),
            ModelCheckpoint(monitor="val_loss", filename='{epoch}-{val_loss:.2f}'),
            best_checkpoint_callback
        ],
        logger=logger,
        profiler='simple',
        auto_lr_find=True,
    )
    
    trainer.tune(net, datamodule=dm)

    trainer.fit(net, datamodule=dm)

    trainer.validate(datamodule=dm, ckpt_path="best")
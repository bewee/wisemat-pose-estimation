import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
from lib.datamodule import *
from lib.layers import Flatten
from lib.F import mpjpe, pcp, pck
from lib.transforms import *


class Net(pl.LightningModule):
    def __init__(self, learning_rate=0.0002, in_channels=2):
        super().__init__()
        self.learning_rate = learning_rate
        self.net = nn.Sequential(
            nn.Dropout(p=0.21093740280701387),
            nn.Conv2d(in_channels, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(5, 5),
            nn.Dropout(p=0.04039388469612025),
            nn.Conv2d(128, 128, 2),
            nn.ReLU(),
            nn.Dropout(p=0.036542891208126344),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Dropout(p=0.018930763782454747),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(),
            nn.Dropout(p=0.03793207249374335),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2048, 26)
        )

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = mpjpe(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
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
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch", "monitor": "val_loss"}]
    
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
    dm = PressurePoseDataModule(input_transform=Stack([MinMaxNormalize(), Boolean()]))
    net = Net()

    dm.setup()
    net.example_input_array = next(iter(dm.val_dataloader()))[0]
    print(ModelSummary(net, max_depth=-1))

    logger = TensorBoardLogger('tb_logs', name='simple-cnn', log_graph=True)

    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, min_delta=0.02),
            ModelCheckpoint(),
            ModelCheckpoint(monitor="val_loss", filename='{epoch}-{val_loss:.2f}'),
            ModelCheckpoint(monitor="val_loss", filename="best")
        ],
        logger=logger,
        profiler='simple',
        auto_lr_find=True,
    )

    trainer.tune(net, datamodule=dm)

    trainer.fit(net, datamodule=dm)

    trainer.validate(datamodule=dm, ckpt_path="best")

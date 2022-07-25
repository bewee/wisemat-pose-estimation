import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
from lib.datamodule import PressurePoseDataModule
from lib.layers import DoubleConv, Down, Up, OutConv
from lib.F import mpjpe, pcp, pck
from lib.transforms import *


class Net(pl.LightningModule):
    def __init__(self, learning_rate=0.0002):
        super().__init__()
        self.learning_rate = learning_rate
        bilinear = True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 13)
        self.loss_function = nn.MSELoss()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.forward(x)
        
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.forward(x)
        
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss)

        self.log("mse", nn.MSELoss()(y_hat, y))
        y_hat_skeletons = HeatmapsToSkeleton()(y_hat.cpu().reshape(-1, 64, 27)).reshape(-1, 13, 2)
        y_skeletons = HeatmapsToSkeleton()(y.cpu().reshape(-1, 64, 27)).reshape(-1, 13, 2)
        self.log("mpjpe", mpjpe(torch.Tensor(y_hat_skeletons), torch.Tensor(y_skeletons)))
        self.log("pcp", pcp(torch.Tensor(y_hat_skeletons), torch.Tensor(y_skeletons)))
        self.log("pck", pck(torch.Tensor(y_hat_skeletons), torch.Tensor(y_skeletons)))

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
        train_data_file = 'data/data-train.pkl',
        val_data_file = 'data/data-val.pkl',
        test_data_file = 'data/data-test.pkl',
        image_dir = 'data',
        input_format='ref',
        target_format='heatmaps',
        num_workers=8,
        input_transform = Stack([StdMeanNormalize(), Boolean(), HistogramEqualize()])
    )
    net = Net()

    dm.setup()
    net.example_input_array = next(iter(dm.val_dataloader()))[0]
    print(ModelSummary(net, max_depth=-1))

    logger = TensorBoardLogger('tb_logs', name='unet', log_graph=True)

    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        callbacks=[
            EarlyStopping(monitor='mpjpe', patience=10, min_delta=0.02),
            ModelCheckpoint(),
            ModelCheckpoint(monitor="mpjpe", filename='{epoch}-{mpjpe:.2f}'),
            ModelCheckpoint(monitor="mpjpe", filename="best"),
            ModelCheckpoint(monitor="val_loss", filename='{epoch}-{val_loss:.6f}')
        ],
        logger=logger,
        profiler='simple',
        auto_lr_find=True,
    )
    
    trainer.tune(net, datamodule=dm)

    trainer.fit(net, datamodule=dm)

    trainer.validate(datamodule=dm, ckpt_path="best")

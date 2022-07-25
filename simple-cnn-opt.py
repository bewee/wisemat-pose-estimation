import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
from lib.datamodule import PressurePoseDataModule
from lib.layers import Flatten
from lib.F import mpjpe, pcp, pck
from lib.constants import constants
import optuna


class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["lr"]
        
        layers = []
        for layer_num in range(len(config['layers'])):
            layer = config['layers'][layer_num]
            in_channels = config['in_channels'] if layer_num == 0 else config['layers'][layer_num-1]['channels']
            layers += [
                nn.Dropout(p=config['dropout_probability']),
                nn.Conv2d(in_channels, layer['channels'], layer['conv_kernelsize'], layer['conv_stride']),
                nn.ReLU(),
                nn.MaxPool2d(layer['pool_kernelsize']),
            ]
        layers += [
            Flatten(),
            nn.Linear(config['flat_features'], config['out_features'])
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = mpjpe(y_hat, y)
        self.log("train_loss", loss)
        self.log("lr", self.learning_rate)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = mpjpe(y_hat, y)
        self.log("val_loss", loss)

        self.log("mpjpe", loss)
        self.log("pcp", pcp(y_hat, y))
        self.log("pck", pck(y_hat, y))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
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
        
def objective(trial):
    config = dict()
    config['layers'] = []
    config['lr'] = trial.suggest_float("lr", 1e-5, 1e-3)
    config['dropout_probability'] = trial.suggest_float("dropout_probability", 0.0, 0.2)
    config['in_channels'] = 2
    config['out_features'] = constants.JOINTS * 2

    h = constants.SENSORS_Y
    w = constants.SENSORS_X
    for layer_num in range(trial.suggest_int("num_layers", 3, 5)):
        config['layers'] += [dict()]
        layer = config['layers'][layer_num]
        layer['channels'] = trial.suggest_categorical(f"layer{layer_num}_channels", [16, 32, 64, 128, 256])

        layer['conv_kernelsize'] = trial.suggest_int(f"layer{layer_num}_conv_kernelsize", 2, 7)
        allowed_conv_kernelsizes = np.asarray(np.arange(2, min(7, w, h), dtype=np.integer))
        layer['conv_kernelsize'] = allowed_conv_kernelsizes[np.abs(allowed_conv_kernelsizes - layer['conv_kernelsize']).argmin()]
        
        layer['conv_stride'] = trial.suggest_int(f"layer{layer_num}_conv_stride", 1, 7)
        allowed_conv_strides = np.asarray(list(filter(
            lambda x: (h - layer['conv_kernelsize'] + 1) % x == 0 and (w - layer['conv_kernelsize'] + 1) % x == 0 and x <= layer['conv_kernelsize'],
            range(1,7)
        )))
        layer['conv_stride'] = allowed_conv_strides[np.abs(allowed_conv_strides - layer['conv_stride']).argmin()]

        h = (h - layer['conv_kernelsize'] + 1) / layer['conv_stride']
        w = (w - layer['conv_kernelsize'] + 1) / layer['conv_stride']

        layer['pool_kernelsize'] = trial.suggest_int(f"layer{layer_num}_pool_kernelsize", 1, 7)
        allowed_pool_kernelsizes = np.asarray(list(filter(
            lambda x: x <= w and x <= h and (h - x + 1) % x == 0 and (w - x + 1) % x == 0,
            range(1,7)
        )))
        layer['pool_kernelsize'] = allowed_pool_kernelsizes[np.abs(allowed_pool_kernelsizes - layer['pool_kernelsize']).argmin()]

        h = (h - layer['pool_kernelsize'] + 1) / layer['pool_kernelsize']
        w = (w - layer['pool_kernelsize'] + 1) / layer['pool_kernelsize']
    config['flat_features'] = int(w*h*config['layers'][-1]['channels'])

    dm = PressurePoseDataModule()
    net = Net(config)

    logger = TensorBoardLogger('tb_logs', name='simple-cnn-opt', log_graph=True, version=trial.number)

    best_checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="best")
    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        strategy='ddp',
        max_epochs=25,
        callbacks=[
            best_checkpoint_callback,
        ],
        logger=logger,
    )

    trainer.fit(net, datamodule=dm)

    best_net = Net.load_from_checkpoint(best_checkpoint_callback.best_model_path, config=config)
    val_data = next(iter(DataLoader(dm.val_set, batch_size=len(dm.val_set))))
    X = val_data[0]
    y = val_data[1]
    y_hat = best_net(X).detach()

    _mpjpe = mpjpe(y_hat, y)
    _pcp = pcp(y_hat, y)
    _pck = pck(y_hat, y)
    _total_params = sum(p.numel() for p in net.parameters())
    return _mpjpe, _pcp, _pck, _total_params

if __name__ == '__main__':
    study = optuna.create_study(
        'sqlite:///optuna.db',
        study_name='simple-cnn',
        load_if_exists=True,
        directions=['minimize', 'maximize', 'maximize', 'minimize'], 
    )
    study.optimize(objective, n_trials=25)

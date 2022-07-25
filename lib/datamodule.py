from torch.utils.data import DataLoader
from .dataset import PressurePoseDataset
from .transforms import Stack, Boolean, Identity, MinMaxNormalize
import pytorch_lightning as pl

class PressurePoseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_file = 'data/data-train_inplace.pkl',
        val_data_file = 'data/data-val_inplace.pkl',
        test_data_file = 'data/data-test_inplace.pkl',
        image_dir = 'data',
        input_format = 'inplace',
        target_format = 'coordinates',
        input_transform = Stack([MinMaxNormalize(), Boolean()]),
        target_transform = None,
        preprocess_input_transform = True,
        preprocess_target_transform = True,
        batch_size = 256,
        num_workers = 0
    ):
        super().__init__()
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.test_data_file = test_data_file
        self.image_dir = image_dir
        self.input_format = input_format
        self.target_format = target_format
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.preprocess_input_transform = preprocess_input_transform
        self.preprocess_target_transform = preprocess_target_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage = None):
        self.train_set = PressurePoseDataset(
            self.train_data_file,
            self.image_dir,
            input_format = self.input_format,
            target_format = self.target_format,
            input_transform = self.input_transform,
            target_transform = self.target_transform,
            preprocess_input_transform = self.preprocess_input_transform,
            preprocess_target_transform = self.preprocess_target_transform
        )
        self.val_set = PressurePoseDataset(
            self.val_data_file,
            self.image_dir,
            input_format = self.input_format,
            target_format = self.target_format,
            input_transform = self.input_transform,
            target_transform = self.target_transform,
            preprocess_input_transform = self.preprocess_input_transform,
            preprocess_target_transform = self.preprocess_target_transform
        )
        self.test_set = PressurePoseDataset(
            self.test_data_file,
            self.image_dir,
            input_format = self.input_format,
            target_format = self.target_format,
            input_transform = self.input_transform,
            target_transform = self.target_transform,
            preprocess_input_transform = self.preprocess_input_transform,
            preprocess_target_transform = self.preprocess_target_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=0)

    def teardown(self, stage = None):
        pass

class BodiesAtRestDataModule(PressurePoseDataModule):
    def __init__(
        self,
        train_data_file = 'data/bodies-at-rest/data-train_inplace.pkl',
        val_data_file = 'data/bodies-at-rest/data-val_inplace.pkl',
        test_data_file = 'data/bodies-at-rest/data-test_inplace.pkl',
        image_dir = 'data',
        input_format = 'inplace',
        target_format = 'coordinates',
        input_transform = Stack([MinMaxNormalize(), Boolean()]),
        target_transform = None,
        preprocess_input_transform = True,
        preprocess_target_transform = True,
        batch_size = 256,
        num_workers = 0
    ):
        super().__init__(
            train_data_file = train_data_file,
            val_data_file = val_data_file,
            test_data_file = test_data_file,
            image_dir = image_dir,
            input_format = input_format,
            target_format = target_format,
            input_transform = input_transform,
            target_transform = target_transform,
            preprocess_input_transform = preprocess_input_transform,
            preprocess_target_transform = preprocess_target_transform,
            batch_size = batch_size,
            num_workers = num_workers
        )

class SLPDataModule(PressurePoseDataModule):
    def __init__(
        self,
        train_data_file = 'data/SLP/data-train_inplace.pkl',
        val_data_file = 'data/SLP/data-val_inplace.pkl',
        test_data_file = 'data/SLP/data-test_inplace.pkl',
        image_dir = 'data',
        input_format = 'inplace',
        target_format = 'coordinates',
        input_transform = Stack([MinMaxNormalize(), Boolean()]),
        target_transform = None,
        preprocess_input_transform = True,
        preprocess_target_transform = True,
        batch_size = 256,
        num_workers = 0
    ):
        super().__init__(
            train_data_file = train_data_file,
            val_data_file = val_data_file,
            test_data_file = test_data_file,
            image_dir = image_dir,
            input_format = input_format,
            target_format = target_format,
            input_transform = input_transform,
            target_transform = target_transform,
            preprocess_input_transform = preprocess_input_transform,
            preprocess_target_transform = preprocess_target_transform,
            batch_size = batch_size,
            num_workers = num_workers
        )

class SoftlineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_data_file = 'data/Softline/data-val_inplace.pkl',
        test_data_file = 'data/Softline/data-test_inplace.pkl',
        image_dir = 'data/Softline',
        input_format = 'inplace',
        target_format = 'coordinates',
        input_transform = Stack([MinMaxNormalize(), Boolean()]),
        target_transform = None,
        preprocess_input_transform = True,
        preprocess_target_transform = True,
        batch_size = 256,
        num_workers = 0
    ):
        super().__init__()
        self.val_data_file = val_data_file
        self.test_data_file = test_data_file
        self.image_dir = image_dir
        self.input_format = input_format
        self.target_format = target_format
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.preprocess_input_transform = preprocess_input_transform
        self.preprocess_target_transform = preprocess_target_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage = None):
        self.val_set = PressurePoseDataset(
            self.val_data_file,
            self.image_dir,
            input_format = self.input_format,
            target_format = self.target_format,
            input_transform = self.input_transform,
            target_transform = self.target_transform,
            preprocess_input_transform = self.preprocess_input_transform,
            preprocess_target_transform = self.preprocess_target_transform
        )
        self.test_set = PressurePoseDataset(
            self.test_data_file,
            self.image_dir,
            input_format = self.input_format,
            target_format = self.target_format,
            input_transform = self.input_transform,
            target_transform = self.target_transform,
            preprocess_input_transform = self.preprocess_input_transform,
            preprocess_target_transform = self.preprocess_target_transform
        )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=0)

    def teardown(self, stage = None):
        pass

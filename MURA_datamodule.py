# References:
# Adapted from ViTShapley:
# I. Covert, C. Kim, and S.-I. Lee, "Learning to estimate Shapley values with vision transformers,"
# presented at the International Conference on Learning Representations 2023, Kigali, Rwanda, 2023.

from lightning.pytorch import LightningDataModule
from MURA_dataset import MURADataset
from torch.utils.data import DataLoader, ConcatDataset

class MURADataModule(LightningDataModule):
    def __init__(self, dataset_location, transforms_original, transforms_augment, num_workers, batch_size, body_part_filter=None, test_data_split="test"):
        super().__init__()
        self.dataset_location = dataset_location
        self.transforms_train_original = transforms_original
        self.transforms_train_augment = transforms_augment
        self.transforms_val = transforms_original
        self.transforms_test = transforms_original
        self.test_data_split = test_data_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.body_part_filter = body_part_filter
        self.setup_flag = False

    def setup(self, stage=None):
        if not self.setup_flag:
            self.train_dataset = ConcatDataset([
                self._make_dataset(self.transforms_train_original, index=0),
                self._make_dataset(self.transforms_train_augment,  index=1),
                self._make_dataset(self.transforms_train_augment, index=2)
            ])

            self.val_dataset = MURADataset(
                dataset_location=self.dataset_location,
                transform_params=self.transforms_val,
                body_part_filter=self.body_part_filter,
                split="val",
            )

            self.test_dataset = MURADataset(
                dataset_location=self.dataset_location,
                transform_params=self.transforms_test,
                body_part_filter=self.body_part_filter,
                split=self.test_data_split,
            )

            self.setup_flag = True

    def _make_dataset(self, transforms, index):
        ds_location = self.dataset_location
        if index == 2:
            ds_location = ds_location + "-processed"
        return MURADataset(
            dataset_location=ds_location,
            transform_params=transforms,
            body_part_filter=self.body_part_filter,
            split="train",
        )

    def _make_dataloader(self, dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True,  drop_last=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset,   shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset,  shuffle=False, drop_last=False)

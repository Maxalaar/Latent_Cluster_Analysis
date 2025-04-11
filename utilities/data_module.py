from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Callable, Optional, Tuple
import torch
from torchvision import transforms

class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_class: Callable,
        dataset_directory: str = "./",
        transform: Optional[Callable] = transforms.ToTensor(),
        batch_size: int = 64,
        train_validation_split: Tuple[float, float] = (0.9, 0.1),
        num_workers: int = 4,
        pin_memory: bool = True,
        **dataset_kwargs
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.dataset_directory = dataset_directory
        self.transform = transform
        self.batch_size = batch_size
        self.train_validation_split = train_validation_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_kwargs = dataset_kwargs
        self.shape = None

    def prepare_data(self):
        self.dataset_class(self.dataset_directory, train=True, download=True, **self.dataset_kwargs)
        self.dataset_class(self.dataset_directory, train=False, download=True, **self.dataset_kwargs)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = self.dataset_class(
                self.dataset_directory, train=True, transform=self.transform, **self.dataset_kwargs
            )
            total_len = len(full_train)
            train_len = int(total_len * self.train_validation_split[0])
            val_len = total_len - train_len
            self.train_set, self.val_set = random_split(full_train, [train_len, val_len])

            sample, _ = self.train_set[0]
            if isinstance(sample, torch.Tensor):
                self.shape = sample.shape

        if stage == "test" or stage is None:
            self.test_set = self.dataset_class(
                self.dataset_directory, train=False, transform=self.transform, **self.dataset_kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

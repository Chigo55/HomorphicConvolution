from pathlib import Path
from typing import Literal, overload

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import LowLightDataset, LowLightSample

LowLightDataLoader = DataLoader[LowLightSample]


class LowLightDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        bench_dir: str,
        infer_dir: str,
        image_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_dir: Path = Path(train_dir)
        self.valid_dir: Path = Path(valid_dir)
        self.bench_dir: Path = Path(bench_dir)
        self.infer_dir: Path = Path(infer_dir)

        self.image_size: int = image_size
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_datasets: list[LowLightDataset] = []
        self.valid_datasets: list[LowLightDataset] = []
        self.bench_datasets: list[LowLightDataset] = []
        self.infer_datasets: list[LowLightDataset] = []

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        if stage is None:
            self.train_datasets = self._set_dataset(data_dir=self.train_dir)
            self.valid_datasets = self._set_dataset(data_dir=self.valid_dir)
            self.bench_datasets = self._set_dataset(data_dir=self.bench_dir)
            self.infer_datasets = self._set_dataset(data_dir=self.infer_dir)
        elif stage == "fit":
            self.train_datasets = self._set_dataset(data_dir=self.train_dir)
            self.valid_datasets = self._set_dataset(data_dir=self.valid_dir)
        elif stage == "validate":
            self.valid_datasets = self._set_dataset(data_dir=self.valid_dir)
        elif stage == "test":
            self.bench_datasets = self._set_dataset(data_dir=self.bench_dir)
        elif stage == "predict":
            self.infer_datasets = self._set_dataset(data_dir=self.infer_dir)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _set_dataset(
        self,
        data_dir: Path,
    ) -> list[LowLightDataset]:
        datasets: list[LowLightDataset] = []
        for folder in data_dir.iterdir():
            if folder.is_dir():
                datasets.append(
                    LowLightDataset(
                        path=folder,
                        image_size=self.image_size,
                    )
                )
        return datasets

    @overload
    def _set_dataloader(
        self,
        datasets: list[LowLightDataset],
        concat: Literal[True],
        shuffle: bool = False,
    ) -> LowLightDataLoader: ...

    @overload
    def _set_dataloader(
        self,
        datasets: list[LowLightDataset],
        concat: Literal[False] = False,
        shuffle: bool = False,
    ) -> list[LowLightDataLoader]: ...

    def _set_dataloader(
        self,
        datasets: list[LowLightDataset],
        concat: bool = False,
        shuffle: bool = False,
    ) -> LowLightDataLoader | list[LowLightDataLoader]:
        if concat:
            dataset_concat: ConcatDataset[LowLightSample] = ConcatDataset(
                datasets=datasets,
            )
            dataloader: LowLightDataLoader = DataLoader(
                dataset=dataset_concat,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
            )
            return dataloader
        dataloaders: list[LowLightDataLoader] = []
        for dataset in datasets:
            loader: LowLightDataLoader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
            )
            dataloaders.append(loader)
        return dataloaders

    def train_dataloader(self) -> LowLightDataLoader:
        return self._set_dataloader(
            datasets=self.train_datasets,
            concat=True,
            shuffle=True,
        )

    def val_dataloader(self) -> LowLightDataLoader:
        return self._set_dataloader(
            datasets=self.valid_datasets,
            concat=True,
        )

    def test_dataloader(self) -> list[LowLightDataLoader]:
        return self._set_dataloader(
            datasets=self.bench_datasets,
        )

    def predict_dataloader(self) -> list[LowLightDataLoader]:
        return self._set_dataloader(
            datasets=self.infer_datasets,
        )

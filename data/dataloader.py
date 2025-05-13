import lightning as L

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class CustomDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = Path(path)
        self.transform = transform
        self.data = list(self.path.glob('*.*'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        return self.transform(image)


class CustomDataModule(L.LightningDataModule):
    def __init__(self, train_dir, valid_dir, infer_dir, bench_dir, transform, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dir = Path(train_dir)
        self.valid_dir = Path(valid_dir)
        self.infer_dir = Path(infer_dir)
        self.bench_dir = Path(bench_dir)

        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_datasets = self._set_dataset(path=self.train_dir)
        self.valid_datasets = self._set_dataset(path=self.valid_dir)
        self.bench_datasets = self._set_dataset(path=self.bench_dir)
        self.infer_datasets = self._set_dataset(path=self.infer_dir)

    def _set_dataset(self, path):
        datasets = []
        for folder in path.iterdir():
            if folder.is_dir():
                datasets.append(
                    CustomDataset(
                        path=folder,
                        transform=self.transform
                    )
                )
        return datasets

    def _set_dataloader(self, datasets, concat=False, shuffle=False):
        if concat:
            dataloader = DataLoader(
                dataset=ConcatDataset(datasets=datasets),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True
            )
            return dataloader
        else:
            dataloaders = []
            for dataset in datasets:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
                dataloaders.append(loader)
            return dataloaders

    def train_dataloader(self):
        return self._set_dataloader(datasets=self.train_datasets, concat=True, shuffle=True)

    def val_dataloader(self):
        return self._set_dataloader(datasets=self.valid_datasets, concat=True)

    def test_dataloader(self):
        return self._set_dataloader(datasets=self.bench_datasets)

    def predict_dataloader(self):
        return self._set_dataloader(datasets=self.infer_datasets)

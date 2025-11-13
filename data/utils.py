import random

from pathlib import Path
from typing import Tuple, cast

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

LowLightSample = Tuple[Tensor, Tensor]


class LowLightDataset(Dataset[LowLightSample]):
    def __init__(self, path: str | Path, image_size: int) -> None:
        super().__init__()
        self.path: Path = Path(path)
        self.image_size: int = image_size
        self.transform: transforms.Compose = transforms.Compose(
            transforms=[
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.low_path: Path = self.path / "low"
        self.high_path: Path = self.path / "high"

        self.low_datas: list[Path] = sorted(self.low_path.rglob(pattern="*.*"))
        self.high_datas: list[Path] = sorted(self.high_path.rglob(pattern="*.*"))

    def __len__(self) -> int:
        return len(self.low_datas)

    def __getitem__(self, index: int) -> LowLightSample:
        low_data: Path = self.low_datas[index]
        high_data: Path = self.high_path / low_data.name

        low_image: Image.Image = Image.open(fp=low_data).convert(mode="RGB")
        high_image: Image.Image = Image.open(fp=high_data).convert(mode="RGB")

        low_image, high_image = self._pair_augment(
            low_image=low_image, high_image=high_image
        )

        low_tensor: Tensor = cast(Tensor, self.transform(img=low_image))
        high_tensor: Tensor = cast(Tensor, self.transform(img=high_image))

        return low_tensor, high_tensor

    def _pair_augment(
        self, low_image: Image.Image, high_image: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            low_image = low_image.transpose(Image.FLIP_LEFT_RIGHT)
            high_image = high_image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            low_image = low_image.transpose(Image.FLIP_TOP_BOTTOM)
            high_image = high_image.transpose(Image.FLIP_TOP_BOTTOM)

        return low_image, high_image

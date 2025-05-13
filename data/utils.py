from torchvision import transforms


class DataTransform:
    def __init__(self, image_size=256):
        self.image_size = image_size

        self.transform = self._build_transform()

    def _build_transform(self):
        base = [
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transforms=base)

    def __call__(self, image):
        return self.transform(img=image)

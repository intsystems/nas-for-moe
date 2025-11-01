import random
import torch
from torchvision import datasets, transforms

class DistortedMNIST(datasets.MNIST):

    def __init__(self, *args,
                 custom_transform=None,
                 distortions=None,
                 permutation=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation = permutation
        self.distortions = distortions or []
        self.custom_transform = custom_transform
        self.input_size = 28 * 28
    
    def __len__(self):
        return super().__len__() * 2

    def __getitem__(self, index):
        if index >= super().__len__():
            img, target = super().__getitem__(index - super().__len__())
        else:
            img, target = super().__getitem__(index)

        if self.custom_transform is not None:
            img = self.custom_transform(img)

        if index >= super().__len__():
            distortion = random.choice(self.distortions)
            img = self.apply_distortion(img, distortion)

        return img, target

    def apply_distortion(self, img: torch.Tensor, distortion: str) -> torch.Tensor:
        """
        Apply the specified distortion to the image tensor.
        """
        c, h, w = img.shape
        img = img.clone()

        if distortion == 'zero_rows':
            # Choose a random row index to zero
            row = random.randrange(h)
            img[:, row, :] = 0.0

        elif distortion == 'zero_columns':
            # Choose a random column index to zero
            col = random.randrange(w)
            img[:, :, col] = 0.0

        elif distortion == 'permutation' and self.permutation is not None:
            img_flat = img.view(c, -1)
            
            # Apply permutation to each channel
            img_permuted = img_flat[:, self.permutation]
            
            # Reshape back to original shape: (C, H*W) -> (C, H, W)
            img = img_permuted.view(c, h, w)
        else:
            raise ValueError(f"Unsupported distortion: {distortion}")

        return img

class DistortedCIFAR10(datasets.CIFAR10):

    def __init__(self, *args, custom_transform=None, distortions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distortions = distortions or []
        self.custom_transform = custom_transform
        self.input_size = 32 * 32 * 3  # CIFAR-10 has 32x32 RGB images
    
    def __len__(self):
        return super().__len__() * 2

    def __getitem__(self, index):
        if index >= super().__len__():
            img, target = super().__getitem__(index - super().__len__())
        else:
            img, target = super().__getitem__(index)

        if self.custom_transform is not None:
            img = self.custom_transform(img)

        if index >= super().__len__():
            distortion = random.choice(self.distortions)
            img = self.apply_distortion(img, distortion)

        return img, target

    def apply_distortion(self, img: torch.Tensor, distortion: str) -> torch.Tensor:
        """
        Apply the specified distortion to the image tensor.
        """
        c, h, w = img.shape
        img = img.clone()

        if distortion == 'zero_rows':
            # Choose a random row index to zero
            row = random.randrange(h)
            img[:, row, :] = 0.0

        elif distortion == 'zero_columns':
            # Choose a random column index to zero
            col = random.randrange(w)
            img[:, :, col] = 0.0

        elif distortion == 'noise':
            # Add random noise
            noise = torch.randn_like(img) * 0.1
            img = torch.clamp(img + noise, 0.0, 1.0)

        else:
            raise ValueError(f"Unsupported distortion: {distortion}")

        return img

import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T


class TransformsSimCLR:
    def __init__(self, size, pretrained=True, n_channel=3, validation=False) -> None:
        self.MAX_SIZE = (512, 512)
        self.train_transforms = T.Compose([
            T.ToTensor(),
            T.Lambda(self.constrained_crop),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # Taking the means of the normal distributions of the 3 channels
            # since we are moving to grayscale
            T.Normalize(mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                        std=np.sqrt(
                            (np.array([0.229, 0.224, 0.225])**2).sum()/9).repeat(n_channel)
                        ) if pretrained is True else T.Lambda(lambda x: x)
        ])

        self.validation_transforms = T.Compose([
            T.ToTensor(),
            T.Resize(size=size),
            # Taking the means of the normal distributions of the 3 channels
            # since we are moving to grayscale
            T.Normalize(mean=np.mean([0.485, 0.456, 0.406]).repeat(n_channel),
                        std=np.sqrt(
                            (np.array([0.229, 0.224, 0.225])**2).sum()/9).repeat(n_channel)
                        ) if pretrained is True else T.Lambda(lambda x: x)
        ])

        self.size = size
        self.validation = validation

    def constrained_crop(self, img):
        crop = np.random.randint(
            low=[0, 0, self.bounding_box[3], self.bounding_box[2]], 
            high=[1 + self.bounding_box[1], 1 + self.bounding_box[0], self.MAX_SIZE[1] + 1, self.MAX_SIZE[0] + 1], 
            size=4
        )
        return F.resized_crop(img, crop[0], crop[1], crop[2]-crop[0]+1, crop[3]-crop[1]+1, self.size)

    def __call__(self, x, bb):
        self.bounding_box = bb# * 224/512
        if not self.validation:
            x_i, x_j = self.train_transforms(x), self.train_transforms(x)
            return x_i, x_j
        else:
            return self.validation_transforms(x)
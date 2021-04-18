import os
import torch

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

root = './data'
real = os.path.join(root, 'real')
sparse = os.path.join(root, 'sparse')
dirs = [root, real, sparse]


def make_dataset(root: str) -> list:
    """
    Creates datasets using the real and sparse images, pairing them up by name.
    """
    dataset = []

    real_dir = 'real'
    sparse_dir = 'sparse'

    real_names = sorted(os.listdir(os.path.join(root, real_dir)))

    for sparse_name in sorted(os.listdir(os.path.join(root, sparse_dir))):
        if sparse_name in real_names:
            real_path = os.path.join(root, real_dir, sparse_name)
            sparse_path = os.path.join(root, sparse_dir, sparse_name)

            item = (real_path, sparse_path)

            dataset.append(item)
        else:
            continue

    return dataset


class CustomVisionDataset(VisionDataset):
    """
    Inherits from the VisionDataset class and creates a transformed dataset with the images as pairs.
    """
    def __init__(self,
                 root,
                 loader=default_loader,
                 real_transform=None,
                 sparse_transform=None):
        super().__init__(root,
                         transform=real_transform,
                         target_transform=sparse_transform)

        samples = make_dataset(self.root)

        self.loader = loader
        self.samples = samples

        self.real_samples = [s[1] for s in samples]
        self.sparse_samples = [s[1] for s in samples]

    def __getitem__(self, index):
        real_path, sparse_path = self.samples[index]

        real_sample = self.loader(real_path)
        sparse_sample = self.loader(sparse_path)

        if self.transform is not None:
            real_sample = self.transform(real_sample)
        if self.target_transform is not None:
            sparse_sample = self.transform(sparse_sample)

        return real_sample, sparse_sample

    def __len__(self):
        return len(self.samples)

def get_images(bs, image_size, workers, shuf=True):
    """
    Creates the dataloader and makes a train, testing split
    """
    dataset = make_dataset('./data')

    ## Transformations
    t = transforms.Compose([transforms.Resize(image_size),
                                                           transforms.CenterCrop(image_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    dataset = CustomVisionDataset('./data', real_transform=t, sparse_transform=t)

    ## Train-test split
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=bs, shuffle=shuf, num_workers=workers)
    dataloader_test = DataLoader(test_dataset, batch_size=bs, shuffle=shuf, num_workers=workers)

    return dataloader_train, dataloader_test

from torchvision.datasets.vision import VisionDataset
from dataio.utils import default_loader, get_all_image_paths

class SimpleFolderLoader(VisionDataset):
    """A generic data loader where the samples with extension ext are located in the tree beginning from the root:

        root/folder_1/folder_11/xxx.ext
        root/folder_1/xxy.ext
        .
        .
        .
        root/folder_n/xxz.ext
        root/folder_n/folder_n1/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        image_paths (list): List of absolute image_paths
    """

    def __init__(self, root, loader, transform=None):
        super(SimpleFolderLoader, self).__init__(root, transform=transform)
        image_paths = get_all_image_paths(self.root)
        if len(image_paths) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

        self.loader = loader
        self.image_paths = image_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_path, image).
        """
        path = self.image_paths[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return path, sample

    def __len__(self):
        return len(self.image_paths)

class BurstLoader(SimpleFolderLoader):
    """Fast batched dataloader for bursted datasets"""

    def __init__(self, root, transform=None, loader=default_loader):
        super(BurstLoader, self).__init__(root, loader, transform=transform)
        self.img_paths = self.image_paths
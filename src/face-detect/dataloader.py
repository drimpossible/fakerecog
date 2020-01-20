import os
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from torchvision.datasets.vision import VisionDataset
from utils import default_loader, get_all_image_paths

class VideoReaderPipeline(Pipeline):
    def __init__(self, filename, batch_size, sequence_length, num_threads, device_id):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.reader = ops.VideoReader(device="gpu", filenames=filename, sequence_length=sequence_length, normalized=False, image_type=types.RGB, dtype=types.FLOAT)

    def define_graph(self):
        output = self.reader(name="Reader")
        return output

class DALILoader():
    def __init__(self, filename, batch_size, sequence_length, workers, device_id):
        self.pipeline = VideoReaderPipeline(filename=filename, batch_size=batch_size, sequence_length=sequence_length, num_threads=workers, device_id=device_id)
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         self.epoch_size,
                                                         auto_reset=True, fill_last_batch=False)
    def __len__(self):
        return int(self.epoch_size)
    def __iter__(self):
        return self.dali_iterator.__iter__()


class SimpleFolderLoader(VisionDataset):
    """A generic data loader where the samples are arranged in the following way:

        root/folder_1/xxx.ext
        root/folder_1/xxy.ext
        .
        .
        .
        root/folder_n/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
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
    """Fast batched dataloader for preprocessing of bursted datasets"""

    def __init__(self, root, transform=None, loader=default_loader):
        super(BurstLoader, self).__init__(root, loader, transform=transform)
        self.img_paths = self.image_paths


























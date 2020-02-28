import os
import glob
import numpy as np
import json
from PIL import Image as pil_image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
import random
import json
import torch

flatten = lambda l: [item for sublist in l for item in sublist]


class ImageDataset(Dataset):

    def __init__(self, json_data, split='train',
                 ):
        self.json_data = json.load(open(json_data, 'r'))
        self.split = split
        self.get_file_list()
        self.transforms = xception_default_data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
            'validation': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
            'test': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
        }
        self.int_label = lambda x: 1 if x == 'FAKE' else 0

    def get_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        """
        Expects a dlib face to generate a quadratic bounding box.
        :param face: dlib face class
        :param width: frame width
        :param height: frame height
        :param scale: bounding box size multiplier to get a bigger face region
        :param minsize: set minimum bounding box size
        :return: x, y, bounding_box_size in opencv form
        """
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize:
            if size_bb < minsize:
                size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check for out of bounds, x-y top left corner
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        # Check for too big bb size for given x, y
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return x1, y1, size_bb

    def get_file_list(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        labels = []
        file_list = []
        for listdata in self.json_data:
            try:
                if listdata['split'] == self.split:
                    frames = glob.glob(listdata['frames_path'] + '/frames/*.jpg')
                    frames = sorted(frames)[::3]
                    file_list.extend(frames)
                    labels.extend([listdata['label']] * len(frames))
            except Exception as e:
                print(str(e))

        self.file_list = file_list
        self.labels = labels


    def __getitem__(self, index: int):
        path = self.file_list[index]
        label = self.labels[index]
        label = self.int_label(label)
        image = self.transforms[self.split](pil_image.open(path))
        return image, label

    def __len__(self):
        return len(self.file_list)


class ImageValidation(ImageDataset):

    def __init__(self, data_path, split='val', num_eval=100):
        self.root_dir = data_path
        self.split = split
        self.num_eval = num_eval
        self.get_file_list()
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
            'val': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
            'test': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
        }

    def get_file_list(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        labels = []
        file_list = []
        for listdata in self.json_data:
            try:
                if listdata['split'] == self.split:
                    file_list.append(listdata['frames_path'])
                    labels.append([listdata['label']])
            except Exception as e:
                print(str(e))

        self.file_list = file_list
        self.labels = labels


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        frames = glob.glob(self.file_list[item] + '/frames/*.jpg')
        frames = sorted(frames)[::2][:32]
        label = self.labels[item]
        images = []
        for f in frames:
            images.append(self.transforms[self.split](pil_image.open(f)))
        label = self.int_label(label)
        return torch.stack(images), torch.stack([label] *len(images))
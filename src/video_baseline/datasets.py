import pickle
import os
from os.path import join
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import torch
import gtransforms
import json

'''
Assumes we generated a metafile in json format that is a list of form:
[{'frame folder path': 'full_path_to_the_frames_folder',
  'vid_id': 'atchvjbadsi' (vid_id as a prefix of the original video name),
  'label': (0,1)},
  .
  .
  .
  ]

'''

class VideoFolder(torch.utils.data.Dataset):
    """
    Something-Something dataset based on *frames* extraction
    """

    def __init__(self,
                 file_input,
                 frames_duration,
                 sample_rate=12,
                 is_val=False,
                 k_split=2,
                 sample_split=1,
                 ):
        """
        :param root: data root path
        :param file_input: inputs path
        :param file_labels: labels path
        :param frames_duration: number of frames
        :param multi_crop_test:
        :param sample_rate: FPS
        :param is_test: is_test flag
        :param k_split: number of splits of clips from the video
        :param sample_split: how many frames sub-sample from each clip
        """
        self.in_duration = frames_duration
        self.sample_rate = sample_rate
        self.is_val = is_val
        with open(file_input, 'r') as f:
            self.json_data = json.load(f)
        self.k_clip = k_split
        self.n_clip = sample_split

        # Kinetics, To Be Updated
        # NOTE: Single channel mean/stev (unlike pytorch Imagenet)
        self.img_mean = [114.75, 114.75, 114.75]
        self.img_std = [57.375, 57.375, 57.375]

        # Transformations
        if not self.is_val:
            self.transforms = [
                gtransforms.GroupResize((224, 224)),
                # gtransforms.GroupRandomCrop((224, 224)),
                # gtransforms.GroupRandomHorizontalFlip()
            ]
        else:
            self.transforms = [
                gtransforms.GroupResize((224, 224))
                # gtransforms.GroupCenterCrop(256),
            ]
        self.transforms += [
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(self.img_mean, self.img_std),
            # TODO: Canceled the Loop Padding
            # gtransforms.LoopPad(self.in_duration),
        ]
        self.transforms = Compose(self.transforms)
        self.prepare_data()

    def prepare_data(self):
        """
        This function creates 3 lists: vid_names, labels and frame_cnts
        :return:
        """
        vid_names = []
        frame_cnts = []
        labels = []
        folder_paths = []
        for listdata in self.json_data:
            try:
                vid_names.append(listdata['vid_id'])
                labels.append(listdata['label'])
                frames_path = listdata['frames_path']
                folder_paths.append(frames_path)
                frames = os.listdir(frames_path)
                frame_cnts.append(int(len(frames)))
            except Exception as e:
                print(str(e))

        self.folder_paths = folder_paths
        self.vid_names = vid_names
        self.labels = labels
        self.frame_cnts = frame_cnts

    # todo: might consider to replace it to opencv, should be much faster
    def load_frame(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        # return Image.open(
        #     join(os.path.dirname(self.data_root), 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')
        return Image.open(
            join(vid_name, '%04d.jpg' % (frame_idx + 1))).convert('RGB')


    def sample_single(self, index):
        """
        Choose and Load frames per video
        :param index:
        :return:
        """
        frame_folder = self.folder_paths[index]
        n_frame = self.frame_cnts[index] - 1
        d = self.in_duration * self.sample_rate
        if n_frame > d:
            if not self.is_val:
                # random sample
                offset = np.random.randint(0, n_frame - d)
            else:
                # center crop
                offset = (n_frame - d) // 2
            frame_list = list(range(offset, offset + d, self.sample_rate))
        else:
            # Temporal Augmentation
            if not self.is_val:
                if n_frame - 2 < self.in_duration:
                    # less frames than needed
                    # pos = np.sort(np.random.choice(list(range(n_frame - 2)), self.in_duration, replace=True))
                    pos = np.linspace(0, n_frame - 2, self.in_duration)
                else:
                    pos = np.sort(np.random.choice(list(range(n_frame - 2)), self.in_duration, replace=False))
            else:
                pos = np.linspace(0, n_frame - 2, self.in_duration)
            frame_list = [round(p) for p in pos]

        frames = []

        for fidx in frame_list:
            frames.append(self.load_frame(frame_folder, fidx))
        return frames

    def __getitem__(self, index):
        label = self.labels(index)
        frames = self.sample_single(index)
        frames = self.transforms(frames)
        frames = frames.permute(1, 0, 2, 3)
        return frames, label

    def __len__(self):
        # return len(self.labels)
        return len(self.json_data)

    def unnormalize(self, img, divisor=255):
        """
        The inverse operation of normalization
        Both the input & the output are in the format of BxCxHxW
        """
        for c in range(len(self.img_mean)):
            img[:, c, :, :].mul_(self.img_std[c]).add_(self.img_mean[c])

        return img / divisor

    def img2np(self, img):
        """
        Convert images in torch tensors of BxCxTxHxW [float32] to a numpy array of BxHxWxC [0-255, uint8]
        Take the first frame along temporal dimension
        if C == 1, that dimension is removed
        """
        img = self.unnormalize(img[:, :, 0, :, :], divisor=1).to(torch.uint8).permute(0, 2, 3, 1)
        if img.shape[3] == 1:
            img = img.squeeze(3)
        return img.cpu().numpy()


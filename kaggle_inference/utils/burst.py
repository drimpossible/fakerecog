from torchvision.datasets.vision import VisionDataset
import os
import torch
import sh, shutil
import cv2
import numpy as np
from .detect_utils import PriorBox


def get_all_video_paths(root_dir):
    video_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        video_paths += [dirpath + '/' + f for f in filenames if (f.endswith('.mp4') or f.endswith('.avi'))]
    return video_paths


def get_all_image_paths(root_dir):
    image_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        image_paths += [dirpath + '/' + f for f in filenames if
                        (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
    return image_paths


class InferenceLoader(VisionDataset):
    """A generic data loader where the directory is filled with videos of extension ext from the root:
        test_folder/xxx.ext
        test_folder/xxy.ext
        .
        .
        .
        test_folder/xxz.ext
    Args:
        root (string): Root directory path to videos.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
     Attributes:
        image_paths (list): List of absolute image_paths
    """

    def __init__(self, cfg, root, frame_rate=12, num_frames=16):
        super(InferenceLoader, self).__init__(root)
        self.video_paths = get_all_video_paths(self.root)
        if len(self.video_paths) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))
        self.frame_rate = frame_rate
        self.num_frames = num_frames
        self.cfg = cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_path, image).
        """
        vid_path = self.video_paths[index]
        vid_file = vid_path.split('/')[-1]
        try:
            capture = cv2.VideoCapture(vid_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            i = 0
            for frame_idx in range(int(frame_count)):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    print("Error grabbing frame %d from movie %s" % (frame_idx, path))

                if frame_idx % 5 == 0:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, _ = frame.shape
                        frame = frame.transpose(2, 0, 1)
                        frame = torch.from_numpy(frame).unsqueeze(0)
                        frames = frame if i == 0 else torch.cat((frames, frame), dim=0)
                        i += 1
                        if i >= self.num_frames:
                            break

            capture.release()
            box_scale = torch.Tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0)
            landms_scale = torch.Tensor(
                [width, height, width, height, width, height, width, height, width, height]).unsqueeze(0).unsqueeze(0)

            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()

            return frames, box_scale, landms_scale, priors, vid_file, width, height
        except:
            print("Error grabbing frames from video %s" % (vid_path))
            return torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), vid_file, 1000, 1000

    def __len__(self):
        return len(self.video_paths)


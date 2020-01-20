import os
import glob
import numpy as np
import json
from PIL import Image as pil_image
import cv2
import dlib
from torch.utils.data import Dataset
from torchvision import transforms


DATA_PATH = '/media/joanna/Work/faceforensicsplusplus/'
flatten = lambda l: [item for sublist in l for item in sublist]

''' TODO:
- face detection on the fly
- read what they did in the paper
'''

class FFImageDataset(Dataset):

    def __init__(self, data_path, split='train',
                 ):
        self.root_dir = data_path
        self.split = split
        self.get_file_list()
        self.transforms = xception_default_data_transforms = {
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
        # Face detector
        self.face_detector = dlib.get_frontal_face_detector()

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
        original_images = sorted(glob.glob(self.root_dir + 'original_sequences/youtube/c23/images/' + '*/*.png'))
        fake_images = sorted(glob.glob(self.root_dir + 'manipulated_sequences/Deepfakes/youtube/c23/images/' + '*/*.png'))
        indexes = json.load(open('ffpp/{}.json'.format(self.split), 'r'))
        org_indexes = flatten(indexes)
        self.file_list = [(i, 0) for i in original_images if i.split('/')[-2] in org_indexes]
        self.file_list += [(i, 1) for i in fake_images if i.split('/')[-2][:3] in org_indexes]

    def __getitem__(self, index: int):
        path, label = self.file_list[index]
        image_path = os.path.join(path)
        # image = self.transform(np.asarray(Image.open(image_path)))
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        face = faces[0]
        height, width = image.shape[:2]
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = self.get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
        image = self.transforms[self.split](pil_image.fromarray(cropped_face))
        return image, label

    def __len__(self):
        return len(self.file_list)
from torchvision.datasets.vision import VisionDataset
import os
import torch
import sh, shutil
import cv2
from imutils.video import FileVideoStream
from .detect_utils import PriorBox

def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def burst_video_into_frames(vid_path, burst_dir, frame_rate, type='opencv'):
    """
    - To burst frames in a directory in shared memory.
    - Returns path to directory containing frames for the specific video
    """
    os.makedirs(burst_dir, exist_ok=True)
    target_mask = os.path.join(burst_dir, '%04d.jpg')
    if type == 'ffmpeg':
        try:
            ffmpeg_args = [
                '-i', vid_path,
                '-q:v', str(1),
                '-f', 'image2',
                '-r', frame_rate,
                target_mask,
            ]
            sh.ffmpeg(*ffmpeg_args)
        except Exception as e:
            print(repr(e))
            return 1
    else:
        try:
            vidcap = cv2.VideoCapture(vid_path)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(target_mask % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                count += 1
        except Exception as e:
            print(repr(e))
            return 1
    return 0

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_all_video_paths(root_dir):
    video_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        video_paths += [dirpath+'/'+f for f in filenames if (f.endswith('.mp4') or f.endswith('.avi'))]
    return video_paths



def get_all_image_paths(root_dir):
    image_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        image_paths += [dirpath+'/'+f for f in filenames if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
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

    def __init__(self, cfg, root, loader=None, transform=None, frame_rate=12, num_frames=16):
        super(InferenceLoader, self).__init__(root, transform=transform)
        self.loader = default_loader
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
        #try:
        v_cap = FileVideoStream(vid_path).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(v_len):
            if i % 5 == 0:
                frame = v_cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if i==0:
                    height, width, _ = frame.shape
                if self.transform is not None:
                    frame = self.transform(frame)
                frames.append(frame)
        out = torch.stack(frames[:self.num_frames])
        box_scale = torch.Tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0)
        landms_scale = torch.Tensor([width, height, width, height, width, height, width, height, width, height]).unsqueeze(0).unsqueeze(0)

        priorbox = PriorBox(self.cfg, image_size=(height, width))
        priors = priorbox.forward()

        return out, box_scale, landms_scale, priors, vid_file, width, height
        #except:
        #    return None, None, None, None, vid_file, None, None

    def __len__(self):
        return len(self.video_paths)


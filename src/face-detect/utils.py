import os, sh, random, torch, cv2
from itertools import product as product
import numpy as np
from math import ceil
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode_boxes(pre, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [1, num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(pre[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes

def decode_landms(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [batch_size, num_priors, 10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [batch_size, num_priors, 4]
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
                        ), dim=2)
    return landms

def decode_fourcc(v):
  v = int(v)
  return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def get_metadata(video_path):
    _v = cv2.VideoCapture(video_path)
    width = int(_v.get(CAP_PROP_FRAME_WIDTH))
    height = int(_v.get(CAP_PROP_FRAME_HEIGHT))
    fps = _v.get(CAP_PROP_FPS)
    frame_cnt = int(_v.get(CAP_PROP_FRAME_COUNT))
    fourcc = _v.get(CAP_PROP_FOURCC)
    encoding = decode_fourcc(fourcc)
    return height, width, fps, frame_cnt, encoding


def burst_video_into_frames(vid_path, burst_dir, frame_rate, format='mp4'):
    """
    - To burst frames in a directory in shared memory.
    - Returns path to directory containing frames for the specific video
    """
    vid_name = vid_path.split('/')[-1]
    out_folder = vid_name[:-(len(format)+1)]
    os.makedirs(burst_dir+'/'+out_folder, exist_ok=True)
    target_mask = os.path.join(burst_dir+'/'+out_folder, '%04d.jpg')
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
    return burst_dir+'/'+out_folder+'/'


class ArrayTracker:
    def __init__(self, yname, xname):
        self.yname = yname
        self.xname = xname
        self.arr = []
        self.epoch = []

    def add(self, inp, epoch):
        self.arr.append(inp)
        self.epoch.append(epoch)

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum*1.0 / self.count*1.0

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def profile_onthefly(func):
        def _wrapper(*args, **kw):
            import line_profiler
            from six.moves import cStringIO
            profile = line_profiler.LineProfiler()

            result = profile(func)(*args, **kw)

            file_ = cStringIO()
            profile.print_stats(stream=file_, stripzeros=True)
            file_.seek(0)
            text = file_.read()
            print(text)
            return result
        return _wrapper


def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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


def get_all_image_paths(root_dir):
    image_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        image_paths += [dirpath+'/'+f for f in filenames if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]
    return image_paths


def get_all_video_paths(root_dir):
    image_paths = []
    for (dirpath, _, filenames) in os.walk(root_dir, followlinks=True):
        image_paths += [dirpath+'/'+f for f in filenames if (f.endswith('.mp4'))]
    return image_paths

def get_enlarged_crop(bbox, image, scale=1.3):
    x1, y1, x2, y2 = bbox
    height, width = image.height, image.width
    size_bb = int(max(x2 - x1, y2 - y1) * scale)

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, x1+size_bb, y1+size_bb
import os, sh, random, torch, cv2
import numpy as np
from math import ceil
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

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
    os.makedirs(burst_dir, exist_ok=True)
    target_mask = os.path.join(burst_dir, '%04d.jpg')
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
    return 

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

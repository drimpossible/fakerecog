# !pip install /kaggle/input/codepackage/sh-1.12.14-py2.py3-none-any.whl # Locally installs sh
# !pip install /kaggle/input/codepackage/filterpy-1.1.0/filterpy-1.1.0/
# !tar xvf /kaggle/input/codepackage/ffmpeg_4.1.4.orig.tar.xz
# !sh /kaggle/working/ffmpeg-4.1.4/configure --disable-x86asm
# !make
# !make install

import sys
import torch, time
from torchvision import transforms
from utils import burst, forward, detect_utils
import csv

class Opts(object):
    def __init__(self):
        self.model = 'Mobilenet0.25'
        self.loader_type = 'burst'
        self.batch_size = 32
        self.nms_threshold = 0.4
        #self.lib_dir = '/homes/53/joanna/fakerecog/kaggle_inference/ckpt'
        self.lib_dir = '/kaggle/input/codepackage/ckpt'
        self.frame_rate = 12
        self.num_frames = 16
        self.scale = 1.2
        #self.ckpt = '/homes/53/joanna/fakerecog/kaggle_inference/ckpt/ckpt.pth.tar'
        self.ckpt = '/kaggle/input/codepackage/ckpt/ckpt.pth.tar'
        #self.ckpt = '/home/anarchicorganizer/codepackage/ckpt/ckpt.pth.tar' #'/kaggle/input/codepackage/ckpt/ckpt.pth.tar'
        self.confidence_threshold = 0.75
        self.resize = 1.
        self.workers = 2
        self.seed = 0
        self.max_track_age = 20
        self.min_track_hits = 3
opt = Opts()
assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'
#test_dir = '/bigssd/joanna/test_dfdc'
   
#test_dir = '/media/anarchicorganizer/Emilia/fakerecog/data/dfdc_testing/test_videos/'

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
cfg = detect_utils.cfg_mnet if opt.model == "Mobilenet0.25" else detect_utils.cfg_res50

# Batch size should be fixed to 1 (takes 1 video at a time). opt.num_frames changes number of frames taken per video. opt.frame_rate determines sampling frame rate (of ffmpeg)
loader = torch.utils.data.DataLoader(burst.InferenceLoader(cfg, test_dir, frame_rate=opt.frame_rate, num_frames=opt.num_frames), batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)
batches = len(loader)
print('Num batches {}'.format(batches))

start = time.time()
inf = forward.InferenceForward(opt)
allprobs, videoid = inf.detect(loader)
end = time.time()
print('Time taken for processing 400 videos: ',(end-start))


submission = open('submission.csv', mode='w')
submission = csv.writer(submission, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for (vid, p) in zip(videoid, allprobs):
    submission.writerow([vid[0], p])

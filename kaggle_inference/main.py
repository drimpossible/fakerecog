
# !tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz # File sent on slack. Just point to ffmpeg in this folder to use ffmpeg.
# !pip install sh-1.12.14-py2.py3-none-any.whl # Locally installs sh

# import sys
# sys.path.append('/kaggle/input/utils/')
import torch, time
from torchvision import transforms
from utils import burst, forward, opts, detect_utils

if __name__ == '__main__':
    opt = opts.parse_args()
    print(opt)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    
    #test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/" # Directory containing all test videos on kaggle. Should be constant.
    test_dir = '/media/anarchicorganizer/Emilia/fakerecog/data/dfdc_testing/test_videos/'

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    cfg = detect_utils.cfg_mnet if opt.model == "Mobilenet0.25" else detect_utils.cfg_res50

    # Batch size should be fixed to 1 (takes 1 video at a time). opt.num_frames changes number of frames taken per video. opt.frame_rate determines sampling frame rate (of ffmpeg)
    loader = torch.utils.data.DataLoader(burst.InferenceLoader(cfg, test_dir, frame_rate=opt.frame_rate, num_frames=opt.num_frames, transform=transforms.ToTensor()), batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)
    batches = len(loader)
    
    start = time.time()
    inf = forward.InferenceForward(opt)
    allprobs, videoid = inf.detect(loader)
    end = time.time()
    print('Time taken for processing 400 videos: ',(end-start))


    # submission_df_xception = pd.DataFrame({"filename": test_videos, "label": predictions})
    # submission_df_xception.to_csv("submission.csv", index=False)
    
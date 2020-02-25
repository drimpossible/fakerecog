
# !apt-get install -y ffmpeg
# !pip install sh

import torch
from torchvision import transforms
import gc
from utils import burst, forward, opts, detect_utils

if __name__ == '__main__':
    opt = opts.parse_args()
    print(opt)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    
    #test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
    test_dir = '/media/anarchicorganizer/Emilia/fakerecog/data/dfdc_testing/test_videos/'

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    cfg = detect_utils.cfg_mnet if opt.model == "Mobilenet0.25" else detect_utils.cfg_res50

    # Open the dataset directory, which should have a dataset json, which contains all info there is about that dataset. 
    loader = torch.utils.data.DataLoader(burst.InferenceLoader(cfg, test_dir, frame_rate=opt.frame_rate, num_frames=opt.num_frames, transform=transforms.ToTensor()), batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)
    batches = len(loader)
    

    inf = forward.InferenceForward(opt)
    allprobs = inf.detect(loader)
    


    # submission_df_xception = pd.DataFrame({"filename": test_videos, "label": predictions})
    # submission_df_xception.to_csv("submission.csv", index=False)
    
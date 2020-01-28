from utils import profile_onthefly
import torch
from torchvision.ops.boxes import batched_nms
from utils import check_ffmpeg_exists, get_all_video_paths, burst_video_into_frames


def test_bursting():
    # Test function for benchmarking bursting speed.
    assert(check_ffmpeg_exists()), "FFMPEG not found"
    datadir = '/media/anarchicorganizer/Emilia/fakerecog/data/dfdc_small/'
    burst_dir = '/media/anarchicorganizer/Emilia/fakerecog/data/bursted_dfdc_small/'
    os.makedirs(burst_dir, exist_ok=True)
    vidformat = 'mp4'
    video_paths = get_all_video_paths(datadir)

    for vid_path in video_paths:
        print(vid_path)
        burst_video_into_frames(vid_path=vid_path, burst_dir=burst_dir)


def test_check_bursting()

if __name__ == '__main__':
    # TODO: Write tests for each individual function to check whenever needed.
    #test_func(batch_size=8, num_points=100000)
    #profile_onthefly(test_func)(batch_size=16, num_points=1000000)
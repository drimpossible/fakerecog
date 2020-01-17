import os
import sh

def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


def get_all_videos(folder_name, format_='mp4'):
    assert(format_ in ['mp4','mkv'])
    f = []
    for (dirpath, _, filenames) in os.walk(folder_name):
        f += [dirpath+'/'+f for f in filenames if f.endswith('.' + format_)]
    return f

def burst_video_into_frames(vid_path, burst_dir):
    """
    - To burst frames in a directory in shared memory.
    - Returns path to directory containing frames for the specific video
    """
    target_mask = os.path.join(burst_dir, '%04d.jpg')
    assert(check_ffmpeg_exists()), "FFMPEG not found"
    try:
        ffmpeg_args = [
            '-i', vid_path,
            '-q:v', str(1),
            '-f', 'image2',
            target_mask,
        ]
        sh.ffmpeg(*ffmpeg_args)
    except Exception as e:
        print(repr(e))


if __name__ == '__main__':
    datadir = '/media/anarchicorganizer/Emilia/fakerecog/data/dfdc_small/'
    vidformat = 'mp4'
    videos = get_all_videos(datadir,vidformat)
    for vid_path in videos:
        vid_folder = vid_path[:-(len(vidformat)+1)]
        os.makedirs(vid_folder, exist_ok=True)
        print(vid_path,vid_folder)
        burst_video_into_frames(vid_path=vid_path, burst_dir=vid_folder)
<<<<<<< HEAD
import json, pickle, torch, os, time, shutil
import opts
from detectors.detect import RetinaFaceDetector
from dataio.crop_and_save import visualize_frames
from trackers.track import get_tracks
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata
import logger
=======
import json, pickle, torch, os, time
import opts, logger
from detect import FaceDetector
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata, get_enlarged_crop
from tqdm import tqdm
import shutil
from PIL import Image

>>>>>>> f7bdc22f0525a5dea8e7d1a1f76b5fde030e53e2

def run_pipeline(opt):
    seed_everything(seed=opt.seed)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    device = torch.device('cuda:'+str(opt.gpu_id) if opt.gpu_id >= 0 else 'cpu')
        
    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)

    # Open the dataset directory, which should have a dataset json, which contains all info there is about that dataset. 
    with open(opt.data_dir+'/'+opt.dataset+'/'+'dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data.keys())
    lenvid = len(videof)
    detector = RetinaFaceDetector(opt=opt, logger=console_logger, device=device) 
    current_idx = opt.current_idx # Enables restarting from middle, incase a run fails. 
    console_logger.debug("==> Starting face detection..")

    for idx in range(current_idx, lenvid):
        console_logger.debug('Starting video: '+str(idx)+'/'+str(lenvid))
        video_path = opt.data_dir+'/'+opt.dataset+'/'+videof[idx]
        assert(os.path.isfile(video_path))
        burst_path = '/dev/shm/face_detector_exp/'
        height, width, fps, frames, encoding = get_metadata(video_path=video_path)
        out_path = opt.out_dir+videof[idx][:-4]+'/' # Replace this hardcoded :-4 with format encoding obtained above?
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(burst_path, exist_ok=True)
        
        # If detections are already saved, ignore this.
        if not os.path.isfile(out_path+'detections.pkl'):
            if opt.loader_type == 'burst':
                burst_video_into_frames(vid_path=video_path, burst_dir=burst_path, format='mp4')
                out = detector.detect(video_path=burst_path, height=height, width=width)
            elif opt.loader_type == 'video':
                out = detector.detect(video_path=video_path, height=height, width=width)
            bboxes, landmarks, confidence, frame_ids, paths = out
            #assert(frame_ids.max()==frames-1)
            tracked_out = get_tracks(opt, bboxes, landmarks, confidence, frame_ids)

            # TODO: Add the cropping part over here?
            with open(out_path+'detections.pkl', 'wb') as handle:
               pickle.dump((tracked_out,paths), handle, protocol=pickle.HIGHEST_PROTOCOL)    
               print("Completed saving to: "+out_path+'detections.pkl')

            # profile_onthefly(detector.detect)(video_path=opt.data_dir+'/'+videof[idx], height=height, width=width)
            # for tracks in tracked_out:
            #     tracklet_bboxes,tracklet_landmarks,tracklet_confidence, tracklet_frames,tracklet_smooth_bboxes = tracks 
            #     visualize_frames(paths=paths, bboxes=tracklet_bboxes, confidence=tracklet_confidence, landmarks=tracklet_landmarks, frame_ids=tracklet_frames)
            #     visualize_frames(paths=paths, bboxes=tracklet_smooth_bboxes, confidence=tracklet_confidence, landmarks=tracklet_landmarks, frame_ids=tracklet_frames)
            shutil.rmtree(burst_path)

if __name__ == '__main__':
    opt = opts.parse_args()
    run_pipeline(opt)

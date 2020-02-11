import json, pickle, torch, os, time, shutil
import opts
from detectors.detect import RetinaFaceDetector
from dataio.crop_and_save import visualize_frames, fix_and_crop_bbox_size
from trackers.track import get_tracks
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata
import logger
import gc

def run_pipeline(opt):
    seed_everything(seed=opt.seed)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    device = torch.device('cuda:'+str(opt.gpu_id) if opt.gpu_id >= 0 else 'cpu')
        
    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)

    # Open the dataset directory, which should have a dataset json, which contains all info there is about that dataset. 
    with open(opt.data_dir+'/'+opt.dataset+'/'+'dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data) #.keys())
    lenvid = len(videof)
    detector = RetinaFaceDetector(opt=opt, logger=console_logger, device=device) 
    current_idx = opt.current_idx if opt.current_idx!=0 else opt.worker_id # Enables restarting from middle, incase a run fails. 
    console_logger.debug("==> Starting face detection..")
    burst_path = '/dev/shm/face_detector_exp_'+str(opt.worker_id)+'/'
    shutil.rmtree(burst_path, ignore_errors=True)

    for idx in range(current_idx, lenvid, opt.total_gpus):
        console_logger.debug('Starting video: '+str(idx)+'/'+str(lenvid))
        try:
            video_path = videof[idx]
            print(video_path)
            assert(os.path.isfile(video_path))
        
            height, width, fps, frames, encoding = get_metadata(video_path=video_path)
            out_path = videof[idx][:-4].replace('/dfdc_preview/', '/dfdc_preview_processed/')+'/' # Replace this hardcoded :-4 with format encoding obtained above?
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(burst_path, exist_ok=True)
        
        # If detections are already saved, ignore this.
            if not os.path.isfile(out_path+'detections.pkl'):
                if opt.loader_type == 'burst':
                    burst_video_into_frames(vid_path=video_path, burst_dir=burst_path, frame_rate=opt.frame_rate, format='mp4')
                    out = detector.detect(video_path=burst_path, height=height, width=width)
                elif opt.loader_type == 'video':
                    out = detector.detect(video_path=video_path, height=height, width=width) # Add a frame_rate option here?
                bboxes, landmarks, confidence, frame_ids, paths = out
            #print(frames,frame_ids,bboxes,paths,out_path)
            #assert(frame_ids.max()==frames-1)
                tracked_out = get_tracks(opt, bboxes, landmarks, confidence, frame_ids)
                i = 0
                for tracks in tracked_out:
                    i+=1
                    tracklet_bboxes, tracklet_landmarks, tracklet_confidence, tracklet_frames, tracklet_smooth_bboxes = tracks
                    fix_and_crop_bbox_size(out_dir=out_path, bboxes=tracklet_smooth_bboxes, frame_ids=tracklet_frames, paths=paths, scale=opt.scale, im_w=width, im_h=height, trackid=i, workers=opt.workers)

            # TODO: Add the cropping part over here?
                with open(out_path+'detections.pkl', 'wb') as handle:
                   pickle.dump((tracked_out,paths), handle, protocol=pickle.HIGHEST_PROTOCOL)    
                   console_logger.debug("Completed saving to: "+out_path+'detections.pkl')

            # profile_onthefly(detector.detect)(video_path=opt.data_dir+'/'+videof[idx], height=height, width=width)
            # for tracks in tracked_out:
            #     tracklet_bboxes,tracklet_landmarks,tracklet_confidence, tracklet_frames,tracklet_smooth_bboxes = tracks 
            #     visualize_frames(paths=paths, bboxes=tracklet_bboxes, confidence=tracklet_confidence, landmarks=tracklet_landmarks, frame_ids=tracklet_frames)
            #     visualize_frames(paths=paths, bboxes=tracklet_smooth_bboxes, confidence=tracklet_confidence, landmarks=tracklet_landmarks, frame_ids=tracklet_frames)
                shutil.rmtree(burst_path)
                del out, tracked_out, bboxes, landmarks, confidence, frame_ids, paths
            gc.collect()
        except:
            print(idx)
if __name__ == '__main__':
    opt = opts.parse_args()
    run_pipeline(opt)

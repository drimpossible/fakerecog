import json, pickle, torch, os, time, shutil
import opts, random
from detectors.detect import RetinaFaceDetector
from trackers.track import get_tracks
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata
import logger
import gc

if __name__ == '__main__':
    opt = opts.parse_args()
    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)
    seed_everything(seed=opt.seed)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    device = torch.device('cuda:'+str(opt.gpu_id) if opt.gpu_id >= 0 else 'cpu')

    # Open the dataset directory, which should have a dataset json, which contains all info there is about that dataset. 
    with open(opt.data_dir+'/'+opt.dataset+'/'+'processed_dataset.json','r') as f:
        data = json.load(f)
    source_videof = list(data.keys())
    lenvid = len(source_videof)
    current_idx = opt.current_idx if opt.current_idx!=0 else opt.process_id # Enables restarting from middle, incase a run fails.

    videof = {}
    for idx in range(lenvid):
        if not data[source_videof[idx]]['original'] in videof:
            videof[data[source_videof[idx]]['original']] = 1       
    videof = list(videof.keys())
    lenvid = len(videof)
    detector = RetinaFaceDetector(opt=opt, logger=console_logger, device=device) 
    console_logger.debug("==> Starting face detection..")

    for idx in range(current_idx, lenvid, opt.total_processes):
        initt = time.time()
        burst_path = '/dev/shm/face_detector_exp/'+videof[idx][:-4]
        shutil.rmtree(burst_path, ignore_errors=True)

        console_logger.debug('Starting video: '+str(idx)+'/'+str(lenvid-1))
        video_path = opt.data_dir+'/'+opt.dataset+'/'+videof[idx]
        assert(os.path.isfile(video_path))
        
        height, width = data[videof[idx]]['height'], data[videof[idx]]['width']
        out_path = opt.out_dir+videof[idx][:-4]+'/' # Replace this hardcoded :-4 with format encoding obtained above?
        
        burstt, detectt, postproct = initt, initt, initt
        if not os.path.isfile(out_path+'detections.pkl'):
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(burst_path, exist_ok=True)
            if opt.loader_type == 'burst':
                burst_video_into_frames(vid_path=video_path, burst_dir=burst_path, frame_rate=opt.frame_rate, format='mp4')
                burstt = time.time()
                out = detector.detect(video_path=burst_path, height=height, width=width)
            elif opt.loader_type == 'video':
                out = detector.detect(video_path=video_path, height=height, width=width) # Add a frame_rate option here.
            
            bboxes, landmarks, confidence, frame_ids, paths = out
            detectt = time.time()
            try:
                tracked_out = get_tracks(opt, bboxes, landmarks, confidence, frame_ids)
            except RuntimeError:
                console_logger.info("WARNING: No faces found in the video: "+videof[idx])
                continue
            with open(out_path+'detections.pkl', 'wb') as handle:
               pickle.dump((out, tracked_out), handle, protocol=pickle.HIGHEST_PROTOCOL)    
               console_logger.info("Completed saving to: "+out_path+'detections.pkl')
            shutil.rmtree(burst_path, ignore_errors=True)
            postproct = time.time()

        gc.collect()
        totalt = time.time()
        console_logger.info('Total time:{0:.4f}\t Burst:{1:.4f}\t Detect:{2:.4f}\t Track:{3:.4f}\t'.format(totalt-initt, burstt-initt, detectt-burstt, postproct-detectt))
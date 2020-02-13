import json, pickle, torch, os, time, shutil
import opts, random
from detectors.detect import RetinaFaceDetector
from dataio.crop_and_save import visualize_frames, fix_bbox, crop_im, diff_crop_im
from trackers.track import get_tracks
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata
import logger
import gc
from multiprocessing import Process

def run_detector(opt):
    seed_everything(seed=opt.seed)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    device = torch.device('cuda:'+str(opt.gpu_id) if opt.gpu_id >= 0 else 'cpu')
        
    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)

    # Open the dataset directory, which should have a dataset json, which contains all info there is about that dataset. 
    with open(opt.data_dir+'/'+opt.dataset+'/'+'processed_dataset.json','r') as f:
        data = json.load(f)
    
    source_videof = list(data.keys())
    lenvid = len(source_videof)

    videof = []
    for idx in range(lenvid):
        videof.append(data[source_videof[idx]]['original'])
    videof = list(set(videof))
    lenvid = len(videof)

    detector = RetinaFaceDetector(opt=opt, logger=console_logger, device=device) 
    current_idx = opt.current_idx if opt.current_idx!=0 else opt.process_id # Enables restarting from middle, incase a run fails. 
    console_logger.debug("==> Starting face detection..")

    
    for idx in range(current_idx, lenvid, opt.total_processes):
        initt = time.time()
        burst_path = '/dev/shm/face_detector_exp/'+videof[idx][:-4]
        shutil.rmtree(burst_path, ignore_errors=True)

        console_logger.debug('Starting video: '+str(idx)+'/'+str(lenvid))
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
            tracked_out = get_tracks(opt, bboxes, landmarks, confidence, frame_ids)
            
            with open(out_path+'detections.pkl', 'wb') as handle:
               pickle.dump((out, tracked_out), handle, protocol=pickle.HIGHEST_PROTOCOL)    
               console_logger.info("Completed saving to: "+out_path+'detections.pkl')
            shutil.rmtree(burst_path, ignore_errors=True)
            postproct = time.time()
        gc.collect()
        totalt = time.time()
        console_logger.info('Total time:{0:.4f}\t Burst:{1:.4f}\t Detect:{2:.4f}\t Track:{3:.4f}\t'.format(totalt-initt, burstt-initt, detectt-burstt, postproct-detectt))
    shutil.rmtree('/dev/shm/face_detector_exp/', ignore_errors=True)

def crop_video(datadir, dataset, outdir, orig_video_path, video_path, frame_rate, scale, width, height, orig_width, orig_height):
    if not os.path.isdir(outdir+video_path[:-4]+'/frames/'):
        salt = str(int(random.random()*(10**5)))
        burst_path = '/dev/shm/cropping_experiment/'+video_path[:-4]+salt
        orig_burst_path = '/dev/shm/cropping_experiment/'+orig_video_path[:-4]+salt

        os.makedirs(orig_burst_path, exist_ok=True)
        burst_video_into_frames(vid_path=datadir+'/'+dataset+'/'+orig_video_path, burst_dir=orig_burst_path, frame_rate=frame_rate, format='mp4')
        os.makedirs(burst_path, exist_ok=True)
        burst_video_into_frames(vid_path=datadir+'/'+dataset+'/'+video_path, burst_dir=burst_path, frame_rate=frame_rate, format='mp4')   
            
        with open(outdir+orig_video_path[:-4]+'/detections.pkl', 'rb') as handle:
            out, tracked_out = pickle.load(handle)

        # TODO: Load tracked out.
        _, _, _, _, paths = out
        os.makedirs(outdir+video_path[:-4]+'/frames/', exist_ok=True)
        os.makedirs(outdir+video_path[:-4]+'/diff_frames/', exist_ok=True)
        for i,tracks in enumerate(tracked_out):
            tracklet_bboxes, tracklet_landmarks, tracklet_confidence, tracklet_frames, tracklet_smooth_bboxes = tracks
            final_bboxes = fix_bbox(bboxes=tracklet_smooth_bboxes, scale=scale, im_w=width, im_h=height, orig_im_w=orig_width, orig_im_h=orig_height)

            for j in range(final_bboxes.size(0)):
                crop_im(outdir=outdir+orig_video_path[:-4]+'/frames/', burst_path=burst_path, frameno=tracklet_frames[j], trackid=i, bbox=final_bboxes[j,:])
                diff_crop_im(outdir=outdir+orig_video_path[:-4]+'/diff_frames/', orig_burst_path=orig_burst_path, burst_path=burst_path, frameno=tracklet_frames[j], trackid=i, bbox=final_bboxes[j,:])

        shutil.rmtree(orig_burst_path, ignore_errors=True)
        shutil.rmtree(burst_path, ignore_errors=True)

if __name__ == '__main__':
    opt = opts.parse_args()
    run_detector(opt)

    with open(opt.data_dir+'/'+opt.dataset+'/'+'processed_dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data.keys())
    lenvid = len(videof)

    shutil.rmtree('/dev/shm/cropping_experiment/', ignore_errors=True)
    os.makedirs('/dev/shm/cropping_experiment/', exist_ok=True)
    for i in range(len(videof)):
        #Process(target=crop_video, args=(datadir=opt.data_dir, dataset=opt.dataset, outdir=opt.out_dir, orig_video_path=data[videof[i]]['original'], video_path=videof[i], frame_rate=opt.frame_rate, scale=opt.scale, width=data[videof[i]]['width'], height=data[videof[i]]['height'], orig_width=data[data[videof[i]]['original']]['width'], orig_height=data[data[videof[i]]['original']]['height'])).start()
        crop_video(datadir=opt.data_dir, dataset=opt.dataset, outdir=opt.out_dir, orig_video_path=data[videof[i]]['original'], video_path=videof[i], frame_rate=opt.frame_rate, scale=opt.scale, width=data[videof[i]]['width'], height=data[videof[i]]['height'], orig_width=data[data[videof[i]]['original']]['width'], orig_height=data[data[videof[i]]['original']]['height'])
    shutil.rmtree('/dev/shm/cropping_experiment/', ignore_errors=True)
import json, pickle, torch, os, time, shutil
import opts, random
from dataio.crop_and_save import visualize_frames, fix_bbox, crop_im, diff_crop_im
from utils import seed_everything, profile_onthefly, burst_video_into_frames
import logger
from multiprocessing import Pool

def crop_video(console_logger, datadir, dataset, outdir, orig_video_path, video_path, frame_rate, scale, width, height, orig_width, orig_height):  
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

        _, _, _, _, paths = out
        os.makedirs(outdir+video_path[:-4]+'/frames/', exist_ok=True)
        os.makedirs(outdir+video_path[:-4]+'/diff_frames/', exist_ok=True)
        for idx,tracks in enumerate(tracked_out):
            tracklet_bboxes, tracklet_landmarks, tracklet_confidence, tracklet_frames, tracklet_smooth_bboxes = tracks
            final_bboxes = fix_bbox(bboxes=tracklet_smooth_bboxes, scale=scale, im_w=width, im_h=height, orig_im_w=orig_width, orig_im_h=orig_height)
            
            for j in range(final_bboxes.size(0)):
                crop_im(outdir=outdir+video_path[:-4]+'/frames/', burst_path=burst_path, framepath=paths[tracklet_frames[j]], trackid=idx, bbox=final_bboxes[j,:])
                diff_crop_im(outdir=outdir+video_path[:-4]+'/diff_frames/', orig_burst_path=orig_burst_path, burst_path=burst_path, framepath=paths[tracklet_frames[j]], trackid=idx, bbox=final_bboxes[j,:])
        console_logger.info('Cropped and saved frames to '+outdir+video_path[:-4]+'/frames/')
        shutil.rmtree(orig_burst_path, ignore_errors=True)
        shutil.rmtree(burst_path, ignore_errors=True)

def check_video(outdir, orig_video_path, video_path):   
    with open(outdir+orig_video_path[:-4]+'/detections.pkl', 'rb') as handle:
            out, tracked_out = pickle.load(handle)

    total = 0
    for idx,tracks in enumerate(tracked_out):
        _, _, _, tracklet_frames, _ = tracks
        total += tracklet_frames.size(0)
    
    numframes = len([name for name in os.listdir(outdir+video_path[:-4]+'/frames/') if os.path.isfile(outdir+video_path[:-4]+'/frames/'+name)])
    numdiffframes = len([name for name in os.listdir(outdir+video_path[:-4]+'/diff_frames/') if os.path.isfile(outdir+video_path[:-4]+'/diff_frames/'+name)])
    if ((numframes<=total-5) or (numdiffframes<=total-5)
        shutil.rmtree(outdir+video_path[:-4]+'/frames/', ignore_errors=True)
        shutil.rmtree(outdir+video_path[:-4]+'/diff_frames/', ignore_errors=True)

if __name__ == '__main__':
    opt = opts.parse_args()
    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)
    seed_everything(seed=opt.seed)

    with open(opt.data_dir+'/'+opt.dataset+'/'+'processed_dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data.keys())
    lenvid = len(videof)
    os.makedirs('/dev/shm/cropping_experiment/', exist_ok=True)

    for idx in range(opt.process_id, lenvid, opt.total_processes):
        console_logger.debug('Processing video: '+str(idx)+'/'+str(lenvid-1)+'..')
        check_video(outdir=opt.out_dir, orig_video_path=data[videof[idx]]['original'], video_path=videof[idx])
        crop_video(console_logger=console_logger, datadir=opt.data_dir, dataset=opt.dataset, outdir=opt.out_dir, orig_video_path=data[videof[idx]]['original'], video_path=videof[idx], frame_rate=opt.frame_rate, scale=opt.scale, width=data[videof[idx]]['width'], height=data[videof[idx]]['height'], orig_width=data[data[videof[idx]]['original']]['width'], orig_height=data[data[videof[idx]]['original']]['height'])

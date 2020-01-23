import json, pickle, torch, os, time
import opts, logger
from detect import FaceDetector
from utils import seed_everything, profile_onthefly, burst_video_into_frames, get_metadata, get_enlarged_crop
from tqdm import tqdm
import shutil
from PIL import Image


if __name__ == '__main__':
    opt = opts.parse_args()
    seed_everything(seed=opt.seed)
    assert(torch.cuda.is_available()), 'Error: No CUDA-enabled device found!'
    device = torch.device('cuda:'+str(opt.gpu_id))

    if not os.path.isdir(opt.log_dir+'/'+opt.exp_name):
        os.mkdir(opt.log_dir+opt.exp_name)

    console_logger = logger.get_logger(opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info(opt)

    with open(opt.data_dir+'dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data.keys())
    lenvid = len(videof)

    detector = FaceDetector(opt=opt, logger=console_logger, device=device)

    current_idx = opt.current_idx # Fix enables restarting from middle, incase a run fails. 

    console_logger.debug("==> Starting face detection..")

    for idx in tqdm(range(current_idx, lenvid)):
        console_logger.info('Starting video: '+str(idx)+'/'+str(lenvid))

        height, width, fps, frames, encoding = get_metadata(video_path=videof[idx].replace('images', 'videos')+'.mp4')
        if height == 0:
            height, width, fps, frames, encoding = get_metadata(
                video_path=videof[idx].replace('youtube/', '').replace('images', 'videos') + '.mp4')
        video_path = videof[idx]
        # if opt.loader_type == 'burst':
        #     # TODO: Add a condition where if already bursted folder exists, don't burst again.
        #     video_path = burst_video_into_frames(vid_path=opt.data_dir+'/'+videof[idx],
        #                                          burst_dir=opt.burst_dir, frame_rate=opt.frame_rate, format='mp4')
        out = detector.detect(video_path=video_path, height=height, width=width)
        
        # Saves the detections to an output file near the video/ in the video directory.
        savepath = video_path+'/detections.pkl' if opt.loader_type == 'burst' else video_path[:-4]+'_detections.pkl'
        # profile_onthefly(detector.detect)(video_path=opt.data_dir+'/'+videof[idx], height=height, width=width)
        # state, bboxes, landmarks, confidence, frame_ids, paths = out
        # detector.visualize_frames(paths=paths, bboxes=bboxes,  confidence=confidence, landmarks=landmarks, frame_ids=frame_ids)
        # break

        # sort predictions
        state, bboxes, landmarks, confidence, frame_ids, paths = out
        sorted_confidence = [x for _, x in sorted(zip(frame_ids, confidence.numpy()), key=lambda pair: pair[0])]
        sorted_bboxes = [x for _, x in sorted(zip(frame_ids, bboxes.numpy()), key=lambda pair: pair[0])]
        sorted_frames_ids = sorted(frame_ids)

        # crop the face from all the frames
        for bbox, conf, ids in zip(sorted_bboxes, sorted_confidence, sorted_frames_ids):
            im_path = video_path + '/%04d' % ids + '.png'
            new_path = im_path.replace('Work', 'Data')
            if not os.path.exists(new_path):
                image = Image.open(video_path + '/%04d' % ids + '.png')
                bbox = get_enlarged_crop(bbox, image)
                crop = image.crop(bbox)
                # save cropped face
                if not os.path.exists(video_path.replace('Work', 'Data')):
                    os.makedirs(video_path.replace('Work', 'Data'))
            ##    if not os.path.exists(video_path.replace('example', 'example_cropped')): #'Work', 'Data')):
             #       os.makedirs(video_path.replace('example', 'example_cropped'))
                crop.save(new_path)
        # delete full size frame
        shutil.rmtree(video_path)

        with open(savepath, 'wb') as handle:
           pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)    
           print("Completed saving to: "+savepath)

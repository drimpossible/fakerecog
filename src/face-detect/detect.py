import json
import pickle
import time
import torch, os
import opts, logger

import time
from utils import PriorBox, decode_boxes, decode_landms, get_metadata, seed_everything, AverageMeter, profile_onthefly
from retinaface import RetinaFace, load_model
from torchvision.ops.boxes import batched_nms
from dataloader import DALILoader

class FaceDetector():
    def __init__(self, opt, logger, device):
        # Set defaults
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.mean = torch.Tensor([104, 117, 123]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        self.opt = opt
        self.logger = logger
        self.device = device

        # Set network
        self.cfg = opts.cfg_mnet if opt.model == "Mobilenet0.25" else opts.cfg_res50
        self.net = load_model(self.cfg, opt.lib_dir+opt.model+'.pth', device)
        #self.logger.info(self.net)

    def detect(self, video_path, height, width):
        data_time, net_time, postproc_time = AverageMeter(), AverageMeter(), AverageMeter()
        initt = time.time()
        box_scale = torch.Tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0).to(self.device)
        landms_scale = torch.Tensor([width, height, width, height, width, height, width, height, width, height]).unsqueeze(0).unsqueeze(0).to(self.device)

        priorbox = PriorBox(self.cfg, image_size=(height, width))
        priors = priorbox.forward()
        priors = priors.unsqueeze(0).to(self.device)

        loader = DALILoader(filename=video_path, batch_size=self.opt.batch_size, sequence_length=1, workers=opt.workers, device_id=self.opt.gpu_id)
        batches = len(loader)
        
        state = 0
        first = time.time()
        init_time = initt-first
        with torch.no_grad():
            start = time.time()
            for i, inputs in enumerate(loader):
                data = inputs[0]["data"].squeeze(dim=1).permute(0,3,1,2)
                data.sub_(self.mean)
                
                datat = time.time()
                data_time.update(datat - start)
                # Batched forward pass
                loc, conf, landms = self.net(data)
                boxes = decode_boxes(loc, priors, self.cfg['variance'])
                boxes = boxes * box_scale / self.opt.resize
                landms = decode_landms(landms, priors, self.cfg['variance'])
                landms = landms * landms_scale / self.opt.resize

                scores = conf[:, :, 1]
                mask = torch.gt(scores, self.opt.confidence_threshold)
                # Alternative methods in postprocessing
                #mask, boxes, landms, scores = mask.cpu(), boxes.cpu(), landms.cpu(), scores.cpu() 
                #landms, boxes, scores = landms[mask], boxes[mask], scores[mask]
                classes = torch.arange(mask.size(0), device=mask.device).repeat_interleave(mask.sum(dim=1))
                scores, boxes, landms = torch.masked_select(scores, mask), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10)
                
                # Parallel GPU-based NMS
                keep = batched_nms(boxes, scores, classes, self.opt.nms_threshold)
                landms, boxes, scores, classes = landms[keep, :], boxes[keep, :], scores[keep], classes[keep]
                classes = classes*(i+1)

                bboxes = boxes if i==0 else torch.cat((bboxes, boxes), dim=0)
                landmarks = landms if i==0 else torch.cat((landmarks, landms), dim=0)
                confidence = scores if i==0 else torch.cat((confidence, scores), dim=0)
                frame_ids = classes if i==0 else torch.cat((frame_ids, classes), dim=0)

                start = time.time()
                postproc_time.update(start- datat)
                total = (start-first)*1.0   

        #self.logger.info('Total time:{0:.4f}\t Data:{dt.sum:.4f}\t Net:{nt.sum:.4f}\t PostP:{pt.sum:.4f}\t'.format(total, dt=data_time, nt=net_time, pt=postproc_time))
        return [state, bboxes.cpu(), landmarks.cpu(), confidence.cpu(), frame_ids.cpu()]

    

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

    current_idx = opt.current_idx # Fix to enable restarting from middle, incase a run fails. 

    console_logger.debug("==> Starting face detection..")

    for idx in range(current_idx, lenvid):
        savepath = opt.data_dir+videof[idx][:-4]+'_pytorch.pkl'
        console_logger.info('Starting video: '+str(idx)+'/'+str(lenvid))

        height, width, fps, frames, encoding = get_metadata(video_path=opt.data_dir+'/'+videof[idx])
        #out = detector.detect(video_path=opt.data_dir+'/'+videof[idx], height=height, width=width)
        profile_onthefly(detector.detect)(video_path=opt.data_dir+'/'+videof[idx], height=height, width=width)
        #with open(savepath, 'wb') as handle:
        #    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
        #     print("Completed saving to: "+savepath)
        
        # curr_idx += 1

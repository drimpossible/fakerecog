import time, torch
import torchvision.transforms as transforms 
from detectors.utils import PriorBox, decode_boxes, decode_landms, AverageMeter, cfg_mnet, cfg_res50
from detectors.retinaface import RetinaFace, load_model
from torchvision.ops.boxes import batched_nms
from dataio import burst #dali, burst

class RetinaFaceDetector():
    def __init__(self, opt, logger, device):
        # Set defaults
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.mean = torch.Tensor([104, 117, 123]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        self.opt = opt
        self.logger = logger
        self.device = device

        # Set network
        self.cfg = cfg_mnet if opt.model == "Mobilenet0.25" else cfg_res50
        self.net = load_model(self.cfg, opt.lib_dir+opt.model+'.pth', device)
        #self.logger.info(self.net)

    def detect(self, video_path, height, width):
        
        box_scale = torch.Tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0).to(self.device)
        landms_scale = torch.Tensor([width, height, width, height, width, height, width, height, width, height]).unsqueeze(0).unsqueeze(0).to(self.device)

        priorbox = PriorBox(self.cfg, image_size=(height, width))
        priors = priorbox.forward()
        priors = priors.unsqueeze(0).to(self.device)

        if self.opt.loader_type == 'video':
            loader = dali.VideoLoader(filename=video_path, batch_size=self.opt.batch_size, sequence_length=1, workers=self.opt.workers, device_id=self.opt.gpu_id)
        
        if self.opt.loader_type == 'burst':
            loader = torch.utils.data.DataLoader(burst.BurstLoader(video_path, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[104.0/256.0, 117.0/256.0, 123.0/256.0], std=[1/256.0, 1.0/256.0, 1.0/256.0])])), batch_size=self.opt.batch_size,
            shuffle=False, num_workers=self.opt.workers, pin_memory=True)
        
        batches = len(loader)
        paths = []

        with torch.no_grad():
            for i, inputs in enumerate(loader):
                # Input batch sequentially by frames (along with path of the frames for verifying correctness)
                if self.opt.loader_type == 'video':
                    data = inputs[0]["data"].squeeze(dim=1).permute(0,3,1,2)
                    data.sub_(self.mean)
                elif self.opt.loader_type == 'burst':
                    path, data = inputs
                    data = data.to(self.device, non_blocking=True)

                # Batched forward pass
                loc, conf, landms = self.net(data)
                boxes = decode_boxes(loc, priors, self.cfg['variance'])
                boxes = boxes * box_scale / self.opt.resize
                landms = decode_landms(landms, priors, self.cfg['variance'])
                landms = landms * landms_scale / self.opt.resize

                scores = conf[:, :, 1]
                mask = torch.gt(scores, self.opt.confidence_threshold)

                # Parallelized masking for thresholding confidence
                classes = self.opt.batch_size*i + torch.arange(mask.size(0), device=mask.device).repeat_interleave(mask.sum(dim=1))
                scores, boxes, landms = torch.masked_select(scores, mask), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10)
                
                # Parallel GPU-based NMS
                keep = batched_nms(boxes, scores, classes, self.opt.nms_threshold)
                landms, boxes, scores, classes = landms[keep, :], boxes[keep, :], scores[keep], classes[keep]

                # Arrange by frames
                idx = torch.argsort(classes)
                landms, boxes, scores, classes = landms[idx, :], boxes[idx, :], scores[idx], classes[idx]

                # Saving bbox (Nx4), landmarks (Nx10), confidence (N) and frame id (N). Frame id maps multiple bboxes to a given frame
                bboxes = boxes if i==0 else torch.cat((bboxes, boxes), dim=0)
                landmarks = landms if i==0 else torch.cat((landmarks, landms), dim=0)
                confidence = scores if i==0 else torch.cat((confidence, scores), dim=0)
                frame_ids = classes if i==0 else torch.cat((frame_ids, classes), dim=0)
                if self.opt.loader_type == 'burst': paths += path 

        #self.logger.info('Total time:{0:.4f}\t Data:{dt.sum:.4f}\t Net:{nt.sum:.4f}\t PostP:{pt.sum:.4f}\t'.format(total, dt=data_time, nt=net_time, pt=postproc_time))
        return [bboxes.cpu(), landmarks.cpu(), confidence.cpu(), frame_ids.cpu(), paths]




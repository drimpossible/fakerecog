import time, torch
import torchvision.transforms as transforms 
from .detect_utils import decode_boxes, decode_landms, AverageMeter, cfg_mnet, cfg_res50, fix_bbox
from .retinaface import RetinaFace, load_model
from .track import get_tracks
from .models import model_selection
from torchvision.ops.boxes import batched_nms
from torchvision.ops import roi_align

class InferenceForward():
    def __init__(self, opt):
        # Set defaults
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.det_mean = torch.Tensor([104.0/256.0, 117.0/256.0, 123.0/256.0]).cuda()
        self.det_std=torch.Tensor([1/256.0, 1.0/256.0, 1.0/256.0]).cuda()
        # # Insert recognition network mean and var here, and load recognition net into self.recnet
        self.rec_mean = torch.Tensor([0.5, 0.5, 0.5]).cuda()
        self.rec_std =  torch.Tensor([0.5, 0.5, 0.5]).cuda()
        self.cfg = cfg_mnet if opt.model == "Mobilenet0.25" else cfg_res50
        self.detnet = load_model(self.cfg, opt.lib_dir)
        model, image_size, *_ = model_selection('xception_fulltraining', num_out_classes=2, pretrained=opt.ckpt)
        self.recnet = model.cuda()


    def detect(self, loader):
        with torch.no_grad():
            start = time.time()
            allprobs = []
            videoid = []
            for i, inputs in enumerate(loader):
                images, box_scale, landms_scale, priors, paths, width, height = inputs
                images, box_scale, landms_scale, priors = images.squeeze(0).cuda(non_blocking=True), box_scale.cuda(non_blocking=True), landms_scale.cuda(non_blocking=True), priors.cuda(non_blocking=True)
                im = images.clone()
                # Normalization for detection model
                im.sub_(self.det_mean[None, :, None, None]).div_(self.det_std[None, :, None, None])

                # Batched forward pass
                loc, conf, landms = self.detnet(im)
                boxes = decode_boxes(loc, priors, self.cfg['variance'])
                boxes = boxes * box_scale / self.opt.resize
                landms = decode_landms(landms, priors, self.cfg['variance'])
                landms = landms * landms_scale / self.opt.resize
                scores = conf[:, :, 1]
                mask = torch.gt(scores, self.opt.confidence_threshold)

                # Parallelized masking for thresholding confidence
                classes = torch.arange(mask.size(0), device=mask.device).repeat_interleave(mask.sum(dim=1))
                scores, boxes, landms = torch.masked_select(scores, mask), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10)
                
                # Parallel GPU-based NMS
                keep = batched_nms(boxes, scores, classes, self.opt.nms_threshold)
                landms, boxes, scores, classes = landms[keep, :], boxes[keep, :], scores[keep], classes[keep]

                # Arrange by frames
                idx = torch.argsort(classes)
                landms, boxes, scores, classes = landms[idx, :], boxes[idx, :], scores[idx], classes[idx]
                maxprob = -1
                tracked_out = get_tracks(self.opt, boxes.cpu(), landms.cpu(), scores.cpu(), classes.cpu())

                # Each pass in this loop is 1 track. Make all frame modifications as per needed for video model here. 
                for idx,tracks in enumerate(tracked_out):
                    tracklet_bboxes, tracklet_landmarks, tracklet_confidence, tracklet_frames, tracklet_smooth_bboxes = tracks
                    final_bboxes = fix_bbox(bboxes=tracklet_smooth_bboxes, scale=self.opt.scale, im_w=width, im_h=height) # We do our own standardization first to ensure all frames are of same size, same as training code.
                    bbox_frame = torch.cat((tracklet_frames.float().unsqueeze(1),final_bboxes),dim=1) 
                    #bbox_frame = torch.cat((final_bboxes,tracklet_frames.float().unsqueeze(1)),dim=1) # For RoI align function, concatenate bbox and which frame the bbox belongs to.
                    bbox_frame = bbox_frame.cuda()
                    # TODO: interpolate the bbox predictions across frames, here I select only oframes for which bbox detected
                    cropped_im = roi_align(images, bbox_frame, (299,299)) # RoI align does cropping and resizing part of it. 
                    cropped_im.sub_(self.rec_mean[None, :, None, None]).div_(self.rec_std[None, :, None, None]) #Mean subtract according to the recognition model
                    prob = self.recnet(cropped_im) # Forward pass in recognition model. For n images, prob should be [n,2] dimensional out.
                    prob = torch.nn.functional.softmax(prob, 1)
                    out = float(prob.mean(0)[1]) # Take the probability of being fake here, minor adjustment.
                    if maxprob < out:
                        maxprob = out
                allprobs.append(maxprob)
                videoid.append(paths[0][0].split('/')[-2]+'.mp4')    
        return allprobs, videoid




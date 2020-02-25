import time, torch
import torchvision.transforms as transforms 
from .detect_utils import decode_boxes, decode_landms, AverageMeter, cfg_mnet, cfg_res50, fix_bbox
from .retinaface import RetinaFace, load_model
from .track import get_tracks
from torchvision.ops.boxes import batched_nms

class InferenceForward():
    def __init__(self, opt):
        # Set defaults
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.det_mean=torch.Tensor([104.0/256.0, 117.0/256.0, 123.0/256.0]).cuda()
        self.det_std=torch.Tensor([1/256.0, 1.0/256.0, 1.0/256.0]).cuda()
        #self.rec_mean=
        #self.rec_std=
        # Set network
        self.cfg = cfg_mnet if opt.model == "Mobilenet0.25" else cfg_res50
        name = 'mobilenetV1X0.25_pretrain.tar' if opt.model == "Mobilenet0.25" else 'ResNet50.pth' 
        self.net = load_model(self.cfg, opt.lib_dir)
        #self.logger.info(self.net)

    def detect(self, loader):
        with torch.no_grad():
            start = time.time()
            for i, inputs in enumerate(loader):
                images, box_scale, landms_scale, priors, paths, width, height = inputs
                images, box_scale, landms_scale, priors = images.squeeze(0).cuda(non_blocking=True), box_scale.cuda(non_blocking=True), landms_scale.cuda(non_blocking=True), priors.cuda(non_blocking=True)
                im = images.clone()
                im.sub_(self.det_mean[None, :, None, None]).div_(self.det_std[None, :, None, None])
                #print(images.shape, box_scale.shape, priors.shape, paths)
                # Batched forward pass
                loc, conf, landms = self.net(im)
                #print(im.shape, loc.shape, priors.shape)
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

                tracked_out = get_tracks(self.opt, boxes.cpu(), landms.cpu(), scores.cpu(), classes.cpu())
                for idx,tracks in enumerate(tracked_out):
                    tracklet_bboxes, tracklet_landmarks, tracklet_confidence, tracklet_frames, tracklet_smooth_bboxes = tracks
                    final_bboxes = fix_bbox(bboxes=tracklet_smooth_bboxes, scale=self.opt.scale, im_w=width, im_h=height)
                # crops = roi_align(image, boxes, box_index)
                # crops.sub_(self.rec_mean[None, :, None, None]).div_(self.rec_std[None, :, None, None])

        return 1




import time, torch
import torchvision.transforms as transforms 
from .detect_utils import decode_boxes, decode_landms, AverageMeter, cfg_mnet, cfg_res50, fix_bbox
from .retinaface import RetinaFace, load_model
from .track import get_tracks
from .models import model_selection
from .i3d import Net as Model
from torchvision.ops.boxes import batched_nms
from torchvision.ops import roi_align
import torch.nn.functional as F

class InferenceForward():
    def __init__(self, opt):
        # Set defaults
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.det_mean = torch.Tensor([104.0, 117.0, 123.0]).cuda()
        self.det_std=torch.Tensor([1, 1.0, 1.0]).cuda()

        self.cfg = cfg_mnet if opt.model == "Mobilenet0.25" else cfg_res50
        self.detnet = load_model(self.cfg, opt.lib_dir)
        if self.opt.recmodel_type == 'image':
            # # Insert recognition network mean and var here, and load recognition net into self.recnet
            self.rec_mean = torch.Tensor([0.5 * 256, 0.5 * 256, 0.5 * 256]).cuda()
            self.rec_std = torch.Tensor([0.5 * 256, 0.5 * 256, 0.5 * 256]).cuda()
            model, image_size, *_ = model_selection('xception_fulltraining', num_out_classes=2, pretrained=opt.ckpt)
            self.opt.crop_size = (299, 299)
            self.recnet = model.cuda().eval()
        elif self.opt.recmodel_type == 'video':
            self.opt.crop_size = (256, 256)
            self.rec_mean = torch.Tensor([114.75, 114.75, 114.75]).cuda()
            self.rec_std = torch.Tensor([57.375, 57.375, 57.375]).cuda()
            if type(opt.ckpt) == str:
                model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt)
                self.recnet = model.cuda().eval()
            elif type(opt.ckpt) == list:
                model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt[0])
                self.recnet1 = model.cuda().eval()
                model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt[1])
                self.recnet2 = model.cuda().eval()
                model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt[2])
                self.recnet3 = model.cuda().eval()
                # model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt[3])
                # self.recnet4 = model.cuda().eval()
                # model = Model(num_classes=2, extract_features=False, loss_type='softmax', weights=opt.ckpt[4])
                # self.recnet5 = model.cuda().eval()
            else:
                raise NotImplementedError()
        self.clip_size = 12

    def adjust_prob(self, prob):
        if prob > 0.5:
            prob = prob - 0.06
        elif prob < 0.5:
            prob = prob + 0.17
        return prob

    def get_prob_5crop_3models(self, video):
        video = video.squeeze()
        cropped_im_top_left = video[:, :, :-32, :-32]
        cropped_im_bottom_right = video[:, :, 32:, 32:]
        cropped_im_center = video[:, :, 16:-16, 16:-16]
        cropped_im_one_side = video[:, :, 32:, 32:]
        cropped_im_other_side = video[:, :, 32:, :-32]
        video_input = torch.empty((15, 3, 12, 224, 224))
        import pdb
        pdb.set_trace()
        video_input[0] = cropped_im_top_left[:, :12, :, :]
        video_input[1] = cropped_im_other_side[:, :12, :, :]
        video_input[2] = cropped_im_one_side[:, :12, :, :]
        video_input[3] = cropped_im_center[:, :12, :, :]
        video_input[4] = cropped_im_bottom_right[:, :12, :, :]

        video_input[5] = cropped_im_top_left[:, 12:24, :, :]
        video_input[6] = cropped_im_other_side[:, 12:24, :, :]
        video_input[7] = cropped_im_one_side[:, 12:24, :, :]
        video_input[8] = cropped_im_center[:, 12:24, :, :]
        video_input[9] = cropped_im_bottom_right[:, 12:24, :, :]

        video_input[10] = cropped_im_top_left[:, 24:, :, :]
        video_input[11] = cropped_im_other_side[:, 24:, :, :]
        video_input[12] = cropped_im_one_side[:, 24:, :, :]
        video_input[13] = cropped_im_center[:, 24:, :, :]
        video_input[14] = cropped_im_bottom_right[:, 24:, :, :]
        video_input = video_input.cuda()
        prob1 = self.recnet1(video_input)
        prob1 = (F.softmax(prob1[:5].mean(0)) + F.softmax(prob1[5:10].mean(0)) + F.softmax(prob1[10:].mean(0)))[1]*0.33333
        prob2 = self.recnet2(video_input)
        prob2 = (F.softmax(prob2[:5].mean(0)) + F.softmax(prob2[5:10].mean(0)) + F.softmax(prob2[10:].mean(0)))[1]*0.33333
        prob3 = self.recnet3(video_input)
        prob3 = (F.softmax(prob3[:5].mean(0)) + F.softmax(prob3[5:10].mean(0)) + F.softmax(prob3[10:].mean(0)))[1]*0.33333
        prob = (prob1 + prob2 + prob3) * 0.3333
        return prob


    def get_prob(self, cropped_im):
        if self.opt.recmodel_type == 'image':
            prob = self.recnet(cropped_im)  # Forward pass in recognition model. For n images, prob should be [n,2] dimensional out.
            prob = torch.nn.functional.softmax(prob, 1)
            out = float(prob.mean(0)[1])  # Take the probability of being fake here, minor adjustment.
        elif self.opt.recmodel_type == 'video':
            # if cropped_im.size(0) > 12:
            #     cropped_im = cropped_im[:12, :, :, :]
            video = cropped_im.unsqueeze(0).transpose(1, 2)
            if type(self.opt.ckpt) == str:
                prob = self.recnet(video)
                prob = torch.nn.functional.softmax(prob, 1)
                out = float(prob.mean(0)[1])  # Take the probability of being fake here, minor adjustment.
            else:
                # prob = 0
                # for im in [cropped_im_bottom_right, cropped_im_center, cropped_im_one_side, cropped_im_other_side, cropped_im_top_left]:
                #     for net in [self.recnet1, self.recnet2, self.recnet3]:
                #         video_input = im.unsqueeze(0).transpose(1, 2)
                #         prob += F.softmax(net(video_input), 1)
                # video_input = torch.empty((15, 3, 12, 224, 224))
                # video_input[0] = video[:, :, :12, :, :]
                # # video_input[1] = video[:, :, 6:18, :, :]
                # video_input[2] = video[:, :, 12:24, :, :]
                # # video_input[3] = video[:, :, 18:30, :, :]
                # video_input[4] = video[:, :, 24:, :, :]
                # video_input = video_input.cuda()
                # prob1 = F.softmax(self.recnet1(video_input), 1).mean(0)
                # #    prob1 = self.adjust_prob(prob1[0][1])
                # prob2 = F.softmax(self.recnet2(video_input), 1).mean(0)
                # #    prob2 = self.adjust_prob(prob2[0][1])
                # prob3 = F.softmax(self.recnet3(video_input), 1).mean(0)
                # #    prob3 = self.adjust_prob(prob3[0][1])
                # # prob4 = F.softmax(self.recnet4(video_input), 1)
                # # prob5 = F.softmax(self.recnet5(video_input), 1)
                # prob = (prob1[1] + prob2[1] + prob3[1]) * 0.3333
                # prob = prob * 0.067
                prob = self.get_prob_5crop_3models(video)
                out = float(prob)  # Take the probability of being fake here, minor adjustment.
        return out



    def detect(self, loader):
        with torch.no_grad():
            start = time.time()
            video_pred = {}
            for i, inputs in enumerate(loader):
                images, box_scale, landms_scale, priors, paths, width, height = inputs
                try:
                    images, box_scale, landms_scale, priors = images.squeeze(0).float().cuda(non_blocking=True), box_scale.cuda(non_blocking=True), landms_scale.cuda(non_blocking=True), priors.cuda(non_blocking=True)
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
                        final_bboxes, final_frames = fix_bbox(bboxes=tracklet_smooth_bboxes, frames=tracklet_frames, scale=self.opt.scale, im_w=width, im_h=height, bs=im.size(0)) # We do our own standardization first to ensure all frames are of same size, same as training code.
                        bbox_frame = torch.cat((final_frames.float().unsqueeze(1), final_bboxes),dim=1)
                        bbox_frame = bbox_frame.cuda()
                        cropped_im = roi_align(images, bbox_frame, self.opt.crop_size) # RoI align does cropping and resizing part of it.
                        cropped_im.sub_(self.rec_mean[None, :, None, None]).div_(self.rec_std[None, :, None, None]) #Mean subtract according to the recognition model
                        out = self.get_prob(cropped_im)
                        if maxprob < out:
                            maxprob = out
                    maxprob = min(maxprob, 0.95)
                    maxprob = max(maxprob, 0.05)
                    if maxprob == -1:
                        maxprob = 0.5
                    video_pred[paths[0]] = maxprob
                    print(paths[0], maxprob)
                except:
                    video_pred[paths[0]]=0.5
                    print(paths[0], maxprob)
        return video_pred




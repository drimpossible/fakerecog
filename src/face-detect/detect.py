import time, torch, opts
from utils import PriorBox, decode_boxes, decode_landms, AverageMeter
from retinaface import RetinaFace, load_model
from torchvision.ops.boxes import batched_nms
from dataloader import DALILoader, BurstLoader
import torchvision.transforms as transforms 
import cv2

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

        if self.opt.loader_type == 'video':
            loader = DALILoader(filename=video_path, batch_size=self.opt.batch_size, sequence_length=1, workers=self.opt.workers, device_id=self.opt.gpu_id)
        
        if self.opt.loader_type == 'burst':
            loader = torch.utils.data.DataLoader(BurstLoader(video_path, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[104.0/256.0, 117.0/256.0, 123.0/256.0], std=[1/256.0, 1.0/256.0, 1.0/256.0])])), batch_size=self.opt.batch_size,
            shuffle=False, num_workers=self.opt.workers, pin_memory=True)
        
        batches = len(loader)
        paths = []
        state = 0
        first = time.time()
        init_time = initt-first
        with torch.no_grad():
            start = time.time()
            for i, inputs in enumerate(loader):
                # Input batch sequentially by frames (along with path of the frames for verifying correctness)
                if self.opt.loader_type == 'video':
                    data = inputs[0]["data"].squeeze(dim=1).permute(0,3,1,2)
                    data.sub_(self.mean)
                elif self.opt.loader_type == 'burst':
                    path, data = inputs
                    data = data.to(self.device, non_blocking=True)
                
                datat = time.time()
                data_time.update(datat - start)

                # Batched forward pass
                loc, conf, landms = self.net(data)
                boxes = decode_boxes(loc, priors, self.cfg['variance'])
                boxes = boxes * box_scale / self.opt.resize
                landms = decode_landms(landms, priors, self.cfg['variance'])
                landms = landms * landms_scale / self.opt.resize

                nett = time.time()
                net_time.update(nett - datat)

                scores = conf[:, :, 1]
                mask = torch.gt(scores, self.opt.confidence_threshold)

                # Parallelized masking for thresholding confidence
                classes = torch.arange(mask.size(0), device=mask.device).repeat_interleave(mask.sum(dim=1))
                scores, boxes, landms = torch.masked_select(scores, mask), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10)
                
                # Parallel GPU-based NMS
                keep = batched_nms(boxes, scores, classes, self.opt.nms_threshold)
                landms, boxes, scores, classes = landms[keep, :], boxes[keep, :], scores[keep], classes[keep]
                classes = classes+ self.opt.batch_size*i

                # Saving bbox (Nx4), landmarks (Nx10), confidence (N) and frame id (N). Frame id maps multiple bboxes to a given frame
                bboxes = boxes if i==0 else torch.cat((bboxes, boxes), dim=0)
                landmarks = landms if i==0 else torch.cat((landmarks, landms), dim=0)
                confidence = scores if i==0 else torch.cat((confidence, scores), dim=0)
                frame_ids = classes if i==0 else torch.cat((frame_ids, classes), dim=0)
                if self.opt.loader_type == 'burst': paths += path
                start = time.time()
                postproc_time.update(start- nett)
                total = (start-first)*1.0   

        self.logger.info('Total time:{0:.4f}\t Data:{dt.sum:.4f}\t Net:{nt.sum:.4f}\t PostP:{pt.sum:.4f}\t'.format(total, dt=data_time, nt=net_time, pt=postproc_time))
        return [state, bboxes.cpu(), landmarks.cpu(), confidence.cpu(), frame_ids.cpu(), paths]

    def visualize_frames(self, paths, bboxes, confidence, landmarks, frame_ids):
        for idx in range(len(frame_ids)):
            img_raw = cv2.imread(paths[frame_ids[idx]], cv2.IMREAD_COLOR)
            cv2.rectangle(img_raw, (bboxes[idx][0], bboxes[idx][1]), (bboxes[idx][2], bboxes[idx][3]), (0, 0, 255), 2)
            cx = bboxes[idx][0]
            cy = bboxes[idx][1] + 12
            text = "{:.4f}".format(confidence[idx])
            cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.circle(img_raw, (landmarks[idx][0], landmarks[idx][1]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (landmarks[idx][2], landmarks[idx][3]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (landmarks[idx][4], landmarks[idx][5]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (landmarks[idx][6], landmarks[idx][7]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (landmarks[idx][8], landmarks[idx][9]), 1, (255, 0, 0), 4)
            cv2.imwrite(paths[frame_ids[idx]][:-4]+'_detections_idx'+str(idx)+'.jpg', img_raw)



from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import cv2
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

filename = '/media/anarchicorganizer/Emilia/fakerecog/face-detect/pytorch/test3.mp4'
import torch
import argparse
from utils import PriorBox, py_cpu_nms, decode, decode_landm
import torch.backends.cudnn as cudnn
from retinaface import RetinaFace
import time
from torchvision.ops.boxes import batched_nms
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='/media/anarchicorganizer/Emilia/fakerecog/face-detect/pytorch/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    #check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class VideoReaderPipeline(Pipeline):
    def __init__(self, filename, batch_size, sequence_length, num_threads, device_id):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.reader = ops.VideoReader(device="gpu", filenames=filename, sequence_length=sequence_length, normalized=False, image_type=types.RGB, dtype=types.FLOAT)

    def define_graph(self):
        output = self.reader(name="Reader")
        return output

class DALILoader():
    def __init__(self, filename, batch_size, sequence_length):
        self.pipeline = VideoReaderPipeline(filename=filename,
                                            batch_size=batch_size,
                                            sequence_length=sequence_length,
                                            num_threads=4,
                                            device_id=0)
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data"],
                                                         self.epoch_size,
                                                         auto_reset=True, fill_last_batch=False)
    def __len__(self):
        return int(self.epoch_size)
    def __iter__(self):
        return self.dali_iterator.__iter__()


loader = DALILoader(filename=filename, batch_size=2, sequence_length=1)
batches = len(loader)



torch.set_grad_enabled(False)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, args.trained_model, args.cpu)
net.eval()
print('Finished loading model!')
print(net)
cudnn.benchmark = True
net = net.cuda()
resize = 1

_v = cv2.VideoCapture(filename)
width = int(_v.get(CAP_PROP_FRAME_WIDTH))
height = int(_v.get(CAP_PROP_FRAME_HEIGHT))
fps = _v.get(CAP_PROP_FPS)
frame_cnt = int(_v.get(CAP_PROP_FRAME_COUNT))
fourcc = _v.get(CAP_PROP_FOURCC)

mean = torch.Tensor([104, 117, 123]).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
box_scale = torch.Tensor([width, height, width, height]).unsqueeze(0).unsqueeze(0).cuda()
landms_scale = torch.Tensor([width, height, width, height, width, height, width, height, width, height]).unsqueeze(0).unsqueeze(0).cuda()
resize, conf_thresh = torch.Tensor([resize]).cuda(), torch.Tensor([args.confidence_threshold]).cuda()
print(width,height, fps, frame_cnt, fourcc)

priorbox = PriorBox(cfg, image_size=(height, width))
priors = priorbox.forward()
priors = priors.unsqueeze(0).cuda()

with torch.no_grad():
    for i, inputs in enumerate(loader):
        data = inputs[0]["data"].squeeze(dim=1).permute(0,3,1,2)
        data.sub_(mean)
        loc, conf, landms = net(data)
        boxes = decode(loc, priors, cfg['variance'])
        landms = decode_landm(landms, priors, cfg['variance'])
        boxes = boxes * box_scale / resize
        landms = landms * landms_scale / resize
        scores = conf[:, :, 1]
        mask = torch.gt(scores,conf_thresh)
        landms, boxes, scores = torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(scores, mask)
        classes = torch.arange(data.size(0), device=boxes.device).repeat_interleave(mask.sum(dim=1))
        keep = batched_nms(boxes, scores, classes, args.nms_threshold)
        landms, boxes, scores, classes = landms[keep, :], boxes[keep, :], scores[keep], classes[keep]
        classes = classes*(i+1)
        if i==0:
            bbox_arr, landms_arr, scores_arr, classes_arr = boxes, landms, scores, classes
        else:
            bbox_arr, landms_arr, scores_arr, classes_arr = torch.cat((bbox_arr, boxes), dim=0), torch.cat((landms_arr, landms), dim=0), torch.cat((scores_arr, scores), dim=0), torch.cat((classes_arr, classes), dim=0)
    print(bbox_arr.size(), landms_arr.size(), scores_arr.size(), classes_arr.size())
    #data = data.cuda(non_blocking=True)
    #
        
    #print(data.size())
    #print(data[0,0,:,0])
























import torch
from itertools import product as product
from math import ceil

class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode_boxes(pre, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [1, num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(pre[:, :, 2:] * variances[1])), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes

def decode_landms(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [batch_size, num_priors, 10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [batch_size, num_priors, 4]
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
                        priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
                        ), dim=2)
    return landms

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum*1.0 / self.count*1.0



def fix_bbox(bboxes, frames, scale, im_w, im_h, bs):
    w = (bboxes[:,2] - bboxes[:,0]).abs()
    h = (bboxes[:,3] - bboxes[:,1]).abs()

    cropw, croph = w.mean()*scale, h.mean()*scale
    minx, miny, maxx, maxy = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([im_w]), torch.Tensor([im_h])

    center_x, center_y = (bboxes[:,2] + bboxes[:,0])/2, (bboxes[:,3] + bboxes[:,1])/2
    interp_frames = np.arange(0,bs)

    interp_center_x, interp_center_y = torch.from_numpy(np.interp(interp_frames, frames.numpy(), center_x.numpy())), torch.from_numpy(np.interp(interp_frames, frames.numpy(), center_y.numpy()))
        
    bbox_out = torch.zeros((interp_center_x.size(0),4))
    bbox_out[:,0], bbox_out[:,1], bbox_out[:,2], bbox_out[:,3] = (interp_center_x-(cropw/2)), (interp_center_y-(croph/2)), (interp_center_x+(cropw/2)), (interp_center_y+(croph/2))
    bbox_out[:,0] = torch.where(bbox_out[:,0] > 0, bbox_out[:,0], minx)
    bbox_out[:,1] = torch.where(bbox_out[:,1] > 0, bbox_out[:,1], miny)
    bbox_out[:,2] = torch.where(bbox_out[:,2] > 0, bbox_out[:,2], maxx)
    bbox_out[:,3] = torch.where(bbox_out[:,3] > 0, bbox_out[:,3], maxy)

    return bbox_out

cfg_mnet = {
    'name': 'Mobilenet0.25',
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

cfg_res50 = {
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

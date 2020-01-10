from utils import profile_onthefly
import torch
from torchvision.ops.boxes import batched_nms

def test_func(batch_size, num_points):
    scores, boxes, landms = torch.rand((batch_size,num_points)).cuda(), torch.rand((batch_size,num_points, 4)).cuda(), torch.rand((batch_size,num_points, 10)).cuda()
    mask = torch.gt(scores, 0.98)
    classes = torch.arange(scores.size(0), device=scores.device).repeat_interleave(mask.sum(dim=1))
    landms, boxes, scores = torch.masked_select(landms,mask.unsqueeze(2)).view(-1,10), torch.masked_select(boxes,mask.unsqueeze(2)).view(-1,4), torch.masked_select(scores, mask)
    # classes = torch.arange(scores.size(0), device=boxes.device).repeat_interleave(mask.sum(dim=1))
    # keep = batched_nms(boxes, scores, classes, 0.4)

if __name__ == '__main__':
    #test_func(batch_size=8, num_points=100000)
    profile_onthefly(test_func)(batch_size=16, num_points=1000000)
from __future__ import division
import collections
import mxnet as mx
import numpy as np
from numpy.linalg import norm
import mxnet.ndarray as nd
import model_zoo
import face_align

__all__ = ['FaceAnalysis',
           'Face']

Face = collections.namedtuple('Face', [
        'bbox', 'landmark', 'det_score'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)

class FaceAnalysis:
    def __init__(self, det_name='retinaface_mnet025_v2'):
        assert det_name is not None
        self.det_model = model_zoo.get_model(det_name)

    def prepare(self, ctx_id, nms=0.4):
        self.det_model.prepare(ctx_id, nms)

    def get(self, img, det_thresh = 0.8, det_scale = 1.0, max_num = 0):
        bboxes, landmarks = self.det_model.detect(img, threshold=det_thresh, scale = det_scale)
        if bboxes.shape[0]==0:
            return []
        if max_num>0 and bboxes.shape[0]>max_num:
            area = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
            img_center = img.shape[0]//2, img.shape[1]//2
            offsets = np.vstack([ (bboxes[:,0]+bboxes[:,2])/2-img_center[1], (bboxes[:,1]+bboxes[:,3])/2-img_center[0] ])
            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
            bindex = np.argmax(area-offset_dist_squared*2.0) # some extra weight on the centering
            bindex = bindex[0:max_num]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]
        
        ret = []
        
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i,4]
            landmark = landmarks[i]
            _img = face_align.norm_crop(img, landmark = landmark)
            face = Face(bbox = bbox, landmark = landmark, det_score = det_score)
            ret.append(face)
        return ret




        
    

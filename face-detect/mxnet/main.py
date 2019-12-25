import json
import pickle
from face_analysis import FaceAnalysis
from video_reader import VideoReader
import time

class ArrayTracker:
    def __init__(self, yname, xname):
        self.yname = yname
        self.xname = xname
        self.arr = []
        self.epoch = []

    def add(self, inp, epoch):
        self.arr.append(inp)
        self.epoch.append(epoch)

if __name__ == '__main__':
    data_dir = '../data/'
    num_id=0
    total=2
        
    model = FaceAnalysis()
    model.prepare(ctx_id=num_id)

    with open(data_dir+'dfdc_small/dataset.json','r') as f:
        data = json.load(f)
    
    videof = list(data.keys())
    lenvid = len(videof)
    curr_idx = 466

    for name in videof[465:]:
        if curr_idx % total == num_id:
            name = name[:-4]
            savepath = data_dir+'dfdc_small/'+name+'.pkl'
            print('Starting video: '+str(curr_idx)+'/'+str(lenvid))

            v= VideoReader(data_dir+'dfdc_small/'+name+'.mp4')
            faces = ArrayTracker(yname='Faces',xname='Timestep')

            for idx, img in enumerate(v):
                face = model.get(img)
                faces.add(face,idx)
        
            assert(len(v)==len(faces.arr) and len(v)==len(faces.epoch))

            with open(savepath, 'wb') as handle:
                pickle.dump(faces, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
            print("Completed saving to: "+savepath)
        
        curr_idx += 1


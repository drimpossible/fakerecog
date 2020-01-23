import os
import pickle
import glob
import json
from PIL import Image
from tqdm import tqdm
import shutil
from utils import get_enlarged_crop

PATH = '/media/joanna/Work/faceforensicsplusplus/'

with open(PATH + 'dataset.json', 'r') as f:
    meta = json.load(f)


for folder in tqdm(meta):
    frames = glob.glob(folder + '/*.png')
    detections = pickle.load(open(folder + '/detections.pkl', 'rb'))
    state, bboxes, landmarks, confidence, frame_ids, paths = detections

    sorted_bboxes = [x for _, x in sorted(zip(frame_ids, bboxes.numpy()), key=lambda pair: pair[0])]
    sorted_confidence = [x for _, x in sorted(zip(frame_ids, confidence.numpy()), key=lambda pair: pair[0])]
    sorted_frames_ids = sorted(frame_ids)


    for bbox, conf, ids in zip(sorted_bboxes, sorted_confidence, sorted_frames_ids):
        im_path = folder + '/%04d' % ids + '.png'
        new_path = im_path.replace('Work', 'Data')
        if not os.path.exists(new_path):
            image = Image.open(folder + '/%04d' % ids + '.png')
            bbox = get_enlarged_crop(bbox, image)
            crop = image.crop(bbox)
            if not os.path.exists(folder.replace('Work', 'Data')):
                os.makedirs(folder.replace('Work', 'Data'))
            crop.save(new_path)
    shutil.rmtree(folder)





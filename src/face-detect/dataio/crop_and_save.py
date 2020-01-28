import os
import pickle
import glob
import json
from PIL import Image
from tqdm import tqdm
import shutil
import cv2
# PATH = '/media/joanna/Work/faceforensicsplusplus/'

# with open(PATH + 'dataset.json', 'r') as f:
#     meta = json.load(f)


def visualize_frames(paths, bboxes, confidence, landmarks, frame_ids):
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
        #print(paths[frame_ids[idx]][:-4]+'_detections_idx'+str(idx)+'.jpg')
        cv2.imwrite(paths[frame_ids[idx]][:-4]+'_detections_idx'+str(idx)+'.jpg', img_raw)

def get_enlarged_crop(bbox, image, scale=1.3):
    x1, y1, x2, y2 = bbox
    height, width = image.height, image.width
    size_bb = int(max(x2 - x1, y2 - y1) * scale)

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, x1+size_bb, y1+size_bb


# for folder in tqdm(meta):
#     frames = glob.glob(folder + '/*.png')
#     detections = pickle.load(open(folder + '/detections.pkl', 'rb'))

#     state, bboxes, landmarks, confidence, frame_ids, paths = detections

#     sorted_bboxes = [x for _, x in sorted(zip(frame_ids, bboxes.numpy()), key=lambda pair: pair[0])]
#     sorted_confidence = [x for _, x in sorted(zip(frame_ids, confidence.numpy()), key=lambda pair: pair[0])]
#     sorted_frames_ids = sorted(frame_ids)

#     for bbox, conf, ids in zip(sorted_bboxes, sorted_confidence, sorted_frames_ids):
#         im_path = folder + '/%04d' % ids + '.png'
#         new_path = im_path.replace('Work', 'Data')
#         if not os.path.exists(new_path):
#             image = Image.open(folder + '/%04d' % ids + '.png')
#             bbox = get_enlarged_crop(bbox, image)
#             crop = image.crop(bbox)
#             if not os.path.exists(folder.replace('Work', 'Data')):
#                 os.makedirs(folder.replace('Work', 'Data'))
#             crop.save(new_path)
#     shutil.rmtree(folder)





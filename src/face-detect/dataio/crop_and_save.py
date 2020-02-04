from PIL import Image
import cv2, torch
from multiprocessing import Process

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

def fix_and_crop_bbox_size(out_dir, bboxes, frame_ids, paths, scale, im_w, im_h, trackid, workers):
    w = (bboxes[:,2] - bboxes[:,0]).abs()
    h = (bboxes[:,3] - bboxes[:,1]).abs()
    cropw, croph = w.mean()*scale, h.mean()*scale
    minx, miny, maxx, maxy = torch.Tensor([0]).long(), torch.Tensor([0]).long(), torch.Tensor([im_w]).long(), torch.Tensor([im_h]).long()

    center_x, center_y = (bboxes[:,2] + bboxes[:,0])/2, (bboxes[:,3] + bboxes[:,1])/2
    bbox_out = bboxes.clone()
    bbox_out[:,0], bbox_out[:,1], bbox_out[:,2], bbox_out[:,3] = center_x-cropw/2, center_y-croph/2, center_x+cropw/2, center_y+croph/2
    bbox_out = bbox_out.long()
    bbox_out[:,0] = torch.where(bbox_out[:,0] > 0, bbox_out[:,0], minx)
    bbox_out[:,1] = torch.where(bbox_out[:,1] > 0, bbox_out[:,1], miny)
    bbox_out[:,2] = torch.where(bbox_out[:,2] > 0, bbox_out[:,2], maxx)
    bbox_out[:,3] = torch.where(bbox_out[:,3] > 0, bbox_out[:,3], maxy)

    for i in range(bbox_out.size(0)):
        Process(target=crop_im, args=(out_dir, paths[frame_ids[i]], trackid, bbox_out[i,:])).start()

def crop_im(out_dir, frame_path, trackid, bbox):
    img = Image.open(frame_path)
    crop = img.crop(bbox.numpy()) 
    frameno = 'track'+str(trackid)+'_'+frame_path.split('/')[-1]
    crop.save(out_dir + frameno)
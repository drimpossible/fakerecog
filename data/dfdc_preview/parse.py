import cv2, json
from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                     CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                     CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

def decode_fourcc(v):
  v = int(v)
  return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def get_metadata(video_path):
    _v = cv2.VideoCapture(video_path)
    width = int(_v.get(CAP_PROP_FRAME_WIDTH))
    height = int(_v.get(CAP_PROP_FRAME_HEIGHT))
    fps = _v.get(CAP_PROP_FPS)
    frame_cnt = int(_v.get(CAP_PROP_FRAME_COUNT))
    fourcc = _v.get(CAP_PROP_FOURCC)
    encoding = decode_fourcc(fourcc)
    return height, width, fps, frame_cnt, encoding



if __name__ == '__main__':
    master = {}

    with open('dataset.json','r') as f:
        data = json.load(f)
    
    for key, value in data.items():
        masterval = {}
        masterval['split'] = value['set']
        masterval['augmentations'] = value['augmentations']
        if value['label'] == 'fake':
            masterval['label'] = 'FAKE'
            masterval['original'] = './original_videos/'+value['target_id']+'/'+value['source_video']+'_'+key[-7:]
        else:
            masterval['label'] = 'REAL'
            masterval['original'] = './'+key
            
        # Encode all metadata in json
        height, width, fps, frames, encoding = get_metadata(video_path='./'+key)
        masterval['height'] = height
        masterval['width'] = width
        masterval['fps'] = fps
        masterval['frames'] = frames
        masterval['encoding'] = encoding
        master['./'+key] = masterval
        #print(master, len(master.keys()))

    with open('processed_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(master, f, ensure_ascii=False, indent=4)
        
            

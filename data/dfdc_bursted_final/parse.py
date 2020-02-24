import os
from os.path import isfile, join
import json
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

with open('../dfdc_large/processed_dataset.json','r') as f:
    data = json.load(f)

dic = {}
absval = []

i = sys.argv[1] 
root = 'dfdc_train_part_'+str(i)+'/'
dirlist = os.listdir(root)

for d in dirlist:
    for f in os.listdir(root+d+'/frames/'): 
        if (isfile(join(root+d+'/frames/', f)) and f[-4:]=='.jpg'):
            tempdic = {}
            try:
                im = Image.open(root+d+'/diff_frames/'+f)
                absdiff = np.array(im.getdata()).sum()
            except:
                im = Image.open(root+d+'/frames/'+f)
                absdiff = -1                
            width, height = im.size
            tempdic['video_path'] = root+d+'.mp4'
            tempdic['video_label'] = data[root+d+'.mp4']['label']
            tempdic['split'] = data[root+d+'.mp4']['our_split']
            tempdic['width'] = width
            tempdic['height'] = height
            tempdic['absval'] = int(absdiff)
            absval.append(absdiff)
            dic[root+d+'/frames/'+f] = tempdic

with open('dfdc_train_part_'+str(i)+'/processed_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

absval = np.array(absval)
plt.hist(absval, bins=10000)
plt.xlim([0, 3000000])
plt.ylim([0, 500])
plt.savefig('dfdc_train_part_'+str(i)+'/absval_plot.png', dpi=1200)

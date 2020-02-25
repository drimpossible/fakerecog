import json
from tqdm import tqdm

with open('./processed_dataset.json','r') as f:
    data = json.load(f)

#Check that each fake video has the same size as original video
videof = list(data.keys())
totlen = len(videof)

for i in tqdm(range(totlen)):
    assert(data[videof[i]]['height'] == data[data[videof[i]]['original']]['height'] or data[videof[i]]['height'] == 0 or data[data[videof[i]]['original']]['height'] == 0), 'Error at '+videof[i]+ '; Original: '+data[videof[i]]['original']+'. Orig height: '+str(data[data[videof[i]]['original']]['height'])+' Curr height: '+str(data[videof[i]]['height'])
    assert(data[videof[i]]['width'] == data[data[videof[i]]['original']]['width']  or data[videof[i]]['width'] == 0 or data[data[videof[i]]['original']]['width'] == 0), 'Error at '+videof[i]+ '; Original: '+data[videof[i]]['original']+'. Orig width: '+str(data[data[videof[i]]['original']]['width'])+' Curr width: '+str(data[videof[i]]['width'])
    assert(data[videof[i]]['frames'] == data[data[videof[i]]['original']]['frames']  or data[videof[i]]['frames'] == 0 or data[data[videof[i]]['original']]['frames'] == 0), 'Error at '+videof[i]+ '; Original: '+data[videof[i]]['original']+'. Orig frames: '+str(data[data[videof[i]]['original']]['frames'])+' Curr frames: '+str(data[videof[i]]['frames'])

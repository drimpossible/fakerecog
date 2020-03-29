import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class SimpleFolderLoader(Dataset):
    def __init__(self, root, split, valfolders, transform=None, choose_class=False):
        with open(root+'/processed_dataset.json','r') as f:
            data = json.load(f)
        assert(split in ['train','val'])
        self.root = root
        self.choose_class = choose_class
        image_paths = []
        labels = []
        val_list = ['dfdc_train_part_'+str(f) for f in valfolders]
        train_list = ['dfdc_train_part_'+str(f) for f in range(50) if f not in valfolders]
        
        avoid_list = train_list if split=='val' else val_list

        for k, v in data.items():
            if k.strip().split('/')[0] not in avoid_list:
                if self.choose_class is not False:
                    if self.choose_class == 'REAL' and v['image_label'] == 'REAL':
                        image_paths.append(k)
                        lb = 0
                        labels.append(lb)

                    if self.choose_class == 'FAKE' and v['image_label'] == 'FAKE':
                        image_paths.append(k)
                        lb = 1
                        labels.append(lb)
                else:
                    image_paths.append(k)
                    lb = 1 if v['image_label'] == 'FAKE' else 0
                    labels.append(lb)
        
        self.labels = torch.from_numpy(np.array(labels))        
        self.image_paths = image_paths
        del data

        # Get image list and labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image_path, image).
        """
        path = self.image_paths[index]
        label = self.labels[index]
        
        image = cv2.imread(self.root+'/'+path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.image_paths)

## This code is interesting. After a bit cleaning I can directly input in the code, but probably not now.
# class ImageValidation(ImageDataset):

#     def __init__(self, json_data, split='validation'):
#         self.json_data = json.load(open(json_data, 'r'))
#         self.split = split
#         self.get_file_list()
#         self.transforms = {
#             'train': transforms.Compose([
#                 transforms.Resize((299, 299)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5] * 3, [0.5] * 3)
#             ]),
#             'validation': transforms.Compose([
#                 transforms.Resize((299, 299)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5] * 3, [0.5] * 3)
#             ]),
#             'test': transforms.Compose([
#                 transforms.Resize((299, 299)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5] * 3, [0.5] * 3)
#             ]),
#         }
#         self.int_label = lambda x: 1 if x == 'FAKE' else 0

#     def get_file_list(self):
#         """
#         This function creates 3 lists: vid_names, labels and frame_cnts
#         :return:
#         """
#         labels = []
#         file_list = []
#         for listdata in self.json_data:
#             try:
#                 if listdata['split'] == self.split:
#                     #if 'dfdc_train_part_0' in listdata['frames_path']:
#                     file_list.append(listdata['frames_path'])
#                     labels.append(listdata['label'])
#             except Exception as e:
#                 print(str(e))

#         self.file_list = file_list
#         self.labels = labels

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class BratsAfrica(Dataset):
    def __init__(self, args, files, mode = 'Training',prompt = 'click', plane = False, transforms=None, multimodal=False):

        self.args = args
        self.files = files # array of file names
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transforms = transforms
        self.multimodal = multimodal


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1
        # label = 4   # the class to be segmented

        """Get the images"""

        di = self.transforms(self.files[index])
        modality_idx = 1
        img = di['image'][modality_idx] if not self.multimodal else di['image'] # only use the middle modality (t1c)
        mask = di['label']

        # mask[mask==0] = 0
        # mask[mask!=0] = 1

        # print("mask before", mask.shape) # 3, 384, 384, 155
        # print("image shape", img.shape) # 4, 384, 384, 155
        # need to convert to one channel

        img = img.unsqueeze(0) if not self.multimodal else img
        # mask = torch.tensor(mask)
        mask = torch.clamp(mask,min=0,max=1).int()

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)

        name = self.files[index]["image"][modality_idx].split('/')[-1].split(".npy")[0]
        name = name.replace("-t1c", "")
        
        # print("name", name)

        # remove extension

        # print("image shape", img.shape)

        image_meta_dict = {'filename_or_obj':name}

        # print("final image and mask shape", img.shape, mask.shape)
        
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }


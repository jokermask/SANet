import os
import random
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop = crop
        self.toTensor = transforms.Compose([transforms.ToTensor()]) 
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
        
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        #make img and gt's shape can be devided by 16, for 1/2 cropped and 3 ML in module 
        img = img.crop((0,0,int(img.size[0]/16)*16,int(img.size[1]/16)*16))
        tagert = target[0:img.size[1],0:img.size[0]]

        img_list = [] 
        target_list = []
        
        if self.transform is not None:
            img = self.transform(img)
            img = transforms.ToPILImage()(img).convert('RGB') 
            
        if self.crop == True:
                    
            crop_size = (int(img.size[0]/2),int(img.size[1]/2))

            for i in range(9):
                dx = int(random.randint(0,1)*img.size[0]*1./2)
                dy = int(random.randint(0,1)*img.size[1]*1./2)
                patch = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
                patch = self.toTensor(patch)
                
                print(np.shape(patch))
                print(np.shape(target[dx:crop_size[0]+dx,dy:crop_size[1]+dy])
                img_list.append(patch)
                target_list.append(target[dx:crop_size[0]+dx,dy:crop_size[1]+dy]) #x for verticle y for horizon
                
        

        return img_list,target_list
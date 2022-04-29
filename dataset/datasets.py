import h5py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class sentinel(Dataset):
    
    @staticmethod 
    def open_h5(path,rgb = False):
        with h5py.File(path, 'r') as h5:
            
            labl = torch.from_numpy(h5['train_id'][:,:].astype('long'))
            if rgb:
                #img = torch.from_numpy(h5['raw'][1:4,:,:]) #check correct bands
                img = np.moveaxis(h5['raw'][:,:,:],0,-1)
            else:
                #img =torch.from_numpy(h5['raw'][:,:,:])
                img = np.moveaxis(h5['raw'][:,:,:],0,-1)
            return(img,labl)
    
    def __init__(self, img_transform=None,label_transform=None,padding=None,root_dir=None, ext='*.nc',rgb=False):
        self.root_dir = root_dir #dataset dir
        self.patch_files = glob.glob(os.path.join(self.root_dir, ext))
        self.padding = padding
        self.img_transform = img_transform
        self.label_transform=label_transform 
        self.rgb = rgb
        
    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        x,y = self.open_h5(self.patch_files[idx], self.rgb)
        
        if self.img_transform:
            x = self.img_transform(x)
            
        if self.label_transform:
            y = self.label_transform(y)
            
        if self.padding:
            pad = nn.ZeroPad2d(self.padding)
            x = pad(x)
            y= pad(y)
        return (x.float(),y) #torch.shape[10,256,256] and [256,256]

    
class s2stats(Dataset):
    
    @staticmethod 
    def open_h5(path,rgb = False):
        with h5py.File(path, 'r') as h5:
            img =torch.from_numpy(h5['raw'][:,:,:])   
            return(img)
        
    
    def __init__(self,root_dir=None, ext='*.nc'):
        self.root_dir = root_dir #dataset dir
        self.patch_files = glob.glob(os.path.join(self.root_dir, ext))
        
    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        img = self.open_h5(self.patch_files[idx])
        return (img.type(torch.float32))
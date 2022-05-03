import h5py
import os
import glob
from matplotlib.font_manager import ttfFontProperty
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as tvF
from torch.utils.data import Dataset



class sentinel(Dataset):

    @staticmethod
    # stack image and label to make transforms on both at same time
    def make_cube(h5):
        labl = h5['train_id'][:,:]
        raw = h5['raw'][:,:,:]
        if raw.shape[0]>raw.shape[2]:
            raw = np.moveaxis(raw,0,-1)
        shape = raw.shape
        cube = np.zeros(shape=(shape[0]+1,shape[1],shape[2]))
        cube[0:-1,:,:]= raw
        cube[-1,:,:]=labl
        return(torch.from_numpy(cube))
        
    
    @staticmethod 
    def open_h5(self,path,rgb = False):
        with h5py.File(path, 'r') as h5:
            #print(h5.keys())
            return(self.make_cube(h5))
    
    @staticmethod
    # Apply transformations on cube (images and labels) and on images
    def transform(self,cube):
            # do transformations applied on both img and label (eg rotations/flips)
            if self.transforms:
                self.transforms(cube)

            # separate img and label
            labl = torch.tensor(cube[-1,:,:],dtype=torch.long)
            img = cube[0:-1,:,:]

            # do transformations applied on img only:
            if self.img_transform:
                img = self.img_transform(img)

            # TODO: return RGB chanells only     
            #if self.rgb:
            #   img=img[0:2]
           
            return(img,labl)
    
    def __init__(self,transforms=None, img_transform=None,padding=None,root_dir=None, ext='*.nc',rgb=False):
        self.root_dir = root_dir #dataset dir
        self.patch_files = glob.glob(os.path.join(self.root_dir, ext))
        self.padding = padding
        self.img_transform = img_transform
        self.transforms = transforms
        self.rgb = rgb

    # return length of dataset
    def __len__(self):
        return len(self.patch_files)

    # getitem of dataset 
    def __getitem__(self, idx):
        cube = self.open_h5(self, self.patch_files[idx], self.rgb)
        x,y = self.transform(self,cube)
            
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
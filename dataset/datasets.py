import h5py
import os
import glob
import numpy as np
import torch
from preprocess.class_dict import class_dict
from torch.utils.data import Dataset


#('B4 (665 nm)',
# 'B3 (560 nm)',
# 'B2 (490 nm)',
# 'B8 (842 nm)',
# 'SRB5 (705 nm)',
# 'SRB6 (740 nm)',
# 'SRB7 (783 nm)',
# 'SRB8A (865 nm)',
# 'SRB11 (1610 nm)',
# 'SRB12 (2190 nm)')

   
    
class sentinel(Dataset):
    
    @staticmethod
    def map_train_id(x):
        if x!=0:
            return(np.float32(class_dict[str(x)]['train_id']))
        else:
            return(np.float32(x))
    
    @staticmethod
    # stack image and label to make transforms on both at same time
    def make_cube(raw,labl):
        
        if raw.shape[0]>raw.shape[2]:
            raw=np.moveaxis(raw,0,-1)
        
        labl = torch.unsqueeze(labl,dim=0)
        cube =torch.cat((raw,labl),0)
        
        return(cube)
        
    
    @staticmethod 
    def open_h5(self,idx):
        with h5py.File(self.patch_files[idx], 'r') as h5:
            #labl = torch.from_numpy(h5['train_id'][:,:].astype('float32'))
            labl = torch.from_numpy(self.map_id(h5['class_code'][:,:]))
        
            
            if self.rgb:
                raw = torch.tensor(h5['raw'][0:3,:,:],dtype=torch.float32)
            else:
                raw = torch.tensor(h5['raw'][:,:,:],dtype=torch.float32)
            return(self.make_cube(raw,labl))
          
    
    @staticmethod
    # Apply transformations on cube (images and labels) and on images
    def transform(self,cube):
            # do transformations applied on both img and label (eg rotations/flips)
            if self.transforms:
                self.transforms(cube)

            # separate img and label
            raw = cube[0:-1,:,:].clone().detach()
            labl = cube[-1,:,:].clone().detach()
            labl=labl.long()

            # do transformations applied on img only:
            if self.img_transform:
                raw = self.img_transform(raw)
           
            return(raw,labl)
    
    def __init__(self,transforms=None, img_transform=None,root_dir=None, timeperiod=1,data='train', ext='*.nc',rgb=False):
        self.data=data # one of test, train or val
        self.root_dir = os.path.join(root_dir,'timeperiod{}'.format(timeperiod),self.data)  #dataset dir
        self.timeperiod = timeperiod
        self.patch_files = glob.glob(os.path.join(self.root_dir, ext))
        self.img_transform = img_transform
        self.transforms = transforms
        self.rgb = rgb
        self.map_id = np.vectorize(self.map_train_id)

    # return length of dataset
    def __len__(self):
        return len(self.patch_files)

    # getitem of dataset 
    def __getitem__(self, idx):
        cube = self.open_h5(self,idx)
        raw,labl = self.transform(self,cube)

        return (raw,labl) 
    

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
        return (img,self.patch_files[idx])
    
    

    

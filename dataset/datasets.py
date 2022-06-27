import h5py
import os
import glob
import numpy as np
import torch
from preprocess.classDict import class_dict
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
    # raw = tensor with all spectral bands [bands, height, width] (raw data) 
    def compute_spectral_indices(raw):
        
        #### Transformed Difference Vegetation Index (TDVI) ####
        # NB! Is NOT a normalized index, so the scale of bands 4 and 8
        # needs to be adjusted (factor: 0.0001)
        B4 = torch.mul(raw[0], torch.tensor(0.0001, dtype = torch.float32))
        B8 = torch.mul(raw[3], torch.tensor(0.0001, dtype = torch.float32))
        
        TDVI_num = torch.sub(B8, B4)
        TDVI_denomPart = torch.add(torch.pow(B8, torch.tensor(2, dtype = torch.float32)), B4)
        TDVI_denom = torch.sqrt(torch.add(TDVI_denomPart, torch.tensor(0.5, dtype = torch.float32)))

        TDVI = torch.mul(torch.tensor(1.5, dtype = torch.float32), torch.div(TDVI_num, TDVI_denom))
        
        #### Enhanced Normalized Difference Impervious Surfaces Index (ENDISI) ####
        B2 = raw[2]
        B3 = raw[1]
        B11 = raw[8]
        B12 = raw[9]
        
        MNDWI = torch.div(torch.sub(B3, B11), torch.add(B3, B11))

        alpha_num = torch.mul(torch.tensor(2, dtype = torch.float32), torch.mean(B2))
        alpha_denom = torch.add(torch.mean(torch.div(B11, B12)), torch.mean(torch.pow(MNDWI, torch.tensor(2, dtype = torch.float32))))
        alpha = torch.div(alpha_num, alpha_denom)

        ENDISI_eqPart = torch.add(torch.div(B11, B12), torch.pow(MNDWI, torch.tensor(2, dtype = torch.float32))) 
        ENDISI_num = torch.sub(B2, torch.mul(alpha, ENDISI_eqPart))
        ENDISI_denom = torch.add(B2, torch.mul(alpha, ENDISI_eqPart))
        ENDISI = torch.div(ENDISI_num, ENDISI_denom)

        return(TDVI, ENDISI)
        
    @staticmethod 
    def open_h5(self,idx):
        with h5py.File(self.patch_files[idx], 'r') as h5:
            labl = torch.from_numpy(h5['train_id'][:,:].astype('float32'))
            #labl = torch.from_numpy(self.map_id(h5['class_code'][:,:]))
        
            if self.rgb:
                raw = torch.tensor(h5['raw'][0:3,:,:],dtype=torch.float32)
            # RGB + spectral indicies    
            elif self.rgbsi:
                raw = torch.tensor(h5['raw'][:,:,:],dtype=torch.float32)
                tdvi, endisi = self.compute_spectral_indices(raw)
                RGB = raw[0:3,:,:]  
                raw = torch.cat((RGB, tdvi.unsqueeze(0), endisi.unsqueeze(0)), dim = 0)
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
            labl = torch.from_numpy(h5['train_id'][:,:].astype('float32'))
            return(img,labl)
        
    
    def __init__(self,root_dir=None, ext='*.nc'):
        self.root_dir = root_dir #dataset dir
        self.patch_files = glob.glob(os.path.join(self.root_dir, ext))
        
    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        img,labl = self.open_h5(self.patch_files[idx])
        return (img,labl,self.patch_files[idx])
    
    

    

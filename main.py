from __future__ import print_function
import argparse
from datetime import datetime
#from locale import normalize
#from math import degrees
#from sklearn.preprocessing import KernelCenterer
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import os

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.datasets import sentinel
from model.models import UNET
import numpy as np

from train.utils import train,test,get_conf,get_config,print_cfg

from tqdm import tqdm



#TODO: save cfg
maxs=torch.tensor([3272., 2232., 1638., 5288., 3847.76098633, 4062.0222168, 5027.98193359, 5334.12207031,4946.20849609, 3493.02246094])
mins=torch.tensor([ 0., 0., 0., 0., 0., -0.91460347,  0.,  0.,  0., -0.07313281])
q_hi = torch.tensor([2102.0, 1716.0, 1398.0, 4732.0, 2434.42919921875, 3701.759765625, 4519.2177734375, 4857.7734375, 3799.80322265625, 3008.8935546875])
q_lo = torch.tensor([102.0, 159.0, 107.0, 77.0, 106.98081970214844, 79.00384521484375, 86.18966674804688, 70.40167236328125, 50.571197509765625, 36.95356750488281])
        
def main():
    # get config file
    cfg = get_conf()
    

    # if use_cuda = TRUE, if cuda (GPU) is available and config no_cuda = false  
    use_cuda = not cfg.train.no_cuda and torch.cuda.is_available()
    cfg.train.device = torch.device('cuda' if use_cuda else 'cpu')

    #manual seed
    if cfg.config.manual_seed:
        torch.manual_seed(cfg.config.seed)
      
  
    # if use_cuda => use cuda_kwargs 
    if use_cuda:
        cfg.dataset.train.kwargs.update(cfg.cuda_kwargs)
        cfg.dataset.test.kwargs.update(cfg.cuda_kwargs)
    
    print_cfg(cfg)
    

 
    
    # Global percentile Min-Max normalization, better known as RobustScaler
    # Less sensitive to outliers than traditional Min-Max normalization
    # https://medium.com/@kesarimohan87/data-preprocessing-6c87d27156
    # https://www.geeksforgeeks.org/feature-scaling-part-3
    # minPer = Min percentile (scalar or tensor)
    # maxPer = Max percentile (scalar or tensor)
    # Sample = Image patch

    class pNormalize(object):
    
        def __init__(self,minPer,maxPer):
            self.minPer = minPer
            self.maxPer = maxPer
        
        def __call__(self,sample):
        
            # According to https://github.com/charlotte-pel/temporalCNN
            norm = (sample-self.minPer[:,None,None])/(self.maxPer[:,None,None]-self.minPer[:,None,None])
            return norm
                                                      
    
    class minMaxScale(object):
    
        def __init__(self,mins,maxs):
            self.mins = mins
            self.maxs = maxs
        
        def __call__(self,sample):
             
            xstd = (sample-self.mins[:,None,None])/(self.maxs[:,None,None]-self.mins[:,None,None])
            scaled = xstd*(self.maxs[:,None,None]-self.mins[:,None,None]) + self.mins[:,None,None]
            return scaled
        
   

    ## chose transforms

    
   
    #normalize = transforms.Normalize(
    #        std=std,
    #        mean=mean
    #        )
                                                      
    pNormalize = pNormalize(
        maxPer = q_hi,
        minPer = q_lo
    )
    
    # img_transforms define transforms to apply on img only
    # Random Apply Define random augmentations to apply on img with prob p:
    #TODO: see kernel_size and sigma used in kth-exjobb
    img_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
            transforms.ColorJitter()
            ]),p=0),
        pNormalize
    ])
    
    #img_labl_transforms define random flips/rot to apply on both img and labl with prob p
    imlab_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90)
            ]),p=0.0)
    ]) 
    
   

    # Create datasets for training & validation,
    training_set = sentinel(root_dir=cfg.dataset.train.root, img_transform=pNormalize,transforms=imlab_transforms, rgb = cfg.dataset.rgb)
    validation_set = sentinel(root_dir=cfg.dataset.test.root, img_transform=pNormalize, rgb = cfg.dataset.rgb)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, **cfg.dataset.train.kwargs)
    validation_loader = DataLoader(validation_set, **cfg.dataset.test.kwargs)

    # Report split sizes 
    print('\nTraining set has {} instances'.format(len(training_set)))
    print('\nValidation set has {} instances'.format(len(validation_set)))    
    
    # batch sample
    img, labl = iter(training_loader).next()
    print(torch.any(torch.isinf(img)))
    #vimg, vlabl = iter(validation_loader).next()
    
    #print(vimg.shape,vlabl.shape)
   
    
    model = UNET(in_channels=img.shape[1]).to(cfg.train.device)
    

    optimizer = optim.NAdam(model.parameters(), lr=cfg.train.lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    #writer is for tensorboard
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir = 'runs/test_run_vm_{}'.format(timestamp)
    writer = SummaryWriter(savedir)

    best_vloss = 1_000_000.
        
    #training_loader = tqdm(training_loader)
    
    for epoch in range(1, cfg.train.epochs + 1):

        # Train one epoch
        avg_loss = train(cfg, model, cfg.train.device, training_loader,
                         optimizer, loss_fn, epoch, writer)
    
        # validate
        avg_vloss = test(cfg, model, cfg.train.device, validation_loader, loss_fn)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # (tensorboard) Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch
                           )
        writer.flush()

        # track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(savedir,'saved_model/model_epoch_{}'.format(epoch))
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            
            
            # only saves models better than previous (when indented)
            if cfg.config.save_model:
                torch.save(model.state_dict(), model_path)
                
                
        


if __name__ == '__main__':
    main()

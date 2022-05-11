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
from dataset.utils import pNormalize
from model.models import UNET
import numpy as np

from train.utils import train, test, get_conf, get_config, print_cfg

from tqdm import tqdm

import munch


maxs=torch.tensor([3272., 2232., 1638., 5288., 3847.76098633, 4062.0222168, 5027.98193359, 5334.12207031,4946.20849609, 3493.02246094])
mins=torch.tensor([ 0., 0., 0., 0., 0., -0.91460347,  0.,  0.,  0., -0.07313281])
q_hi = torch.tensor([2102.0, 1716.0, 1398.0, 4732.0, 2434.42919921875, 3701.759765625, 4519.2177734375, 4857.7734375, 3799.80322265625, 3008.8935546875])
q_lo = torch.tensor([102.0, 159.0, 107.0, 77.0, 106.98081970214844, 79.00384521484375, 86.18966674804688, 70.40167236328125, 50.571197509765625, 36.95356750488281])
        
def main():
    # get config file
    cfg = get_conf()
    
    # update unique train kwargs with non unique dataset kwargs 
    cfg.dataset.train_kwargs.update(cfg.dataset.kwargs)
    cfg.dataset.test_kwargs.update(cfg.dataset.kwargs)
    
    # parameters for dataset 
    train_kwargs = cfg.dataset.train_kwargs
    test_kwargs = cfg.dataset.test_kwargs
    
    # parameters for dataloaders 
    loader_train_kwargs=cfg.data_loader.train_kwargs
    loader_test_kwargs =cfg.data_loader.test_kwargs
    
    
    #for testing purpouse 
    dry_run = cfg.config.dry_run 
    # train on GPU if no_cuda is false (and cuda is available) 
    use_cuda = not cfg.config.no_cuda and torch.cuda.is_available()
    # how often (#Batch) to log results in terminal  
    log_intervall=cfg.config.log_intervall 
    # Save model for future inference
    save_model=cfg.config.save_model # Boolean 
      
    # Define savedir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir = 'runs/{band}_bands_{timeperiod}_vm_{timestamp}'.format(band='rgb' if train_kwargs.rgb else 'all',
                                                      timeperiod=train_kwargs.timeperiod,
                                                      timestamp= timestamp)
    
    # load model for inference 
    load_model = cfg.config.load_model # Boolean
    load_path = cfg.config.load_path # path to saved model
    
    # for reproducability 
    manual_seed = cfg.config.manual_seed
    seed = cfg.config.seed
  
    # train on device
    device = torch.device('cuda' if use_cuda else 'cpu')
    epochs = cfg.train.epochs
    lr = cfg.optimizer.lr

    # parameters for dry run
    if dry_run:
        log_intervall=5
        savedir='runs/test_{}.format(timestamp)'
        
    #manual seed
    if manual_seed:
        torch.manual_seed(seed)
      
    # if use_cuda => use cuda_kwargs 
    if use_cuda:
        loader_train_kwargs.update(cfg.cuda_kwargs)
        loader_test_kwargs.update(cfg.cuda_kwargs)
    
    print_cfg(cfg)
    
    #writer is for tensorboard
    writer = SummaryWriter(savedir)
    
    
    # initialize normalizer 
    pNorm = pNormalize(
        maxPer = q_hi,
        minPer = q_lo
    )
    
    # img_transforms define transforms to apply on img only
    # RandomApply Define random augmentations to apply on img with prob p:
    # TODO: see kernel_size and sigma used in kth-exjobb
    img_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
            transforms.ColorJitter()
            ]),p=0.3),
        pNorm
    ])
    
    #imlab_transforms define random flips/rotations to apply on both img and labl with prob p
    imlab_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90)
            ]),p=0.4)
    ]) 
    
   
    # Define datasets for training & validation
    training_set = sentinel( **train_kwargs,
                            img_transform=img_transforms,
                            transforms=imlab_transforms
                           )
    
    validation_set = sentinel(**test_kwargs,
                              img_transform=pNorm 
                             )

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, **loader_train_kwargs)
    validation_loader = DataLoader(validation_set, **loader_test_kwargs)

    # Report split sizes 
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances\n'.format(len(validation_set)))    
    
    # batch sample
    img, labl = iter(training_loader).next()
   
    # Define model
    model = UNET(in_channels=img.shape[1]).to(device)
    
    if load_model: 
        model.load_state_dict(torch.load(load_path))
    
    # specify optimizer
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    
    # specify loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    # initiate best_vloss
    best_vloss = 1_000_000.
        
    #training_loader = tqdm(training_loader)
    
    for i in range(epochs):
        epoch = i+1
        
        if not load_model:
            # Train one epoch
            avg_loss = train(cfg, model, device, training_loader,
                             optimizer, loss_fn, epoch, writer)
    
        # validate / test
        avg_vloss = test(cfg, model, device, validation_loader, loss_fn)

        print('Loss train: {} Loss valid: {}'.format(round(avg_loss,2), round(avg_vloss,2)))

        # (tensorboard) Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch
                           )
        writer.flush()

        # track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            # save model for inference
            if save_model:
                model_path = os.path.join(savedir,'saved_model/model_epoch_{}.pt'.format(epoch))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                torch.save(model.state_dict(), model_path)
                
                
        


if __name__ == '__main__':
    main()
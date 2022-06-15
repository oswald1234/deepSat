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

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.datasets import sentinel
from dataset.stats import quantiles,ce_weights, classCounts

from preprocess.classDict import   class_dict
from dataset.utils import pNormalize, crossEntropyLossWeights, classCount
from train.loss import focalTverskyLoss
from model.models import UNET
import numpy as np
import time

from train.utils import train, test, get_conf, get_config, print_cfg,save_cfg,get_savedir, eval

import munch



def get_transforms(cfg):
    
    
 # initialize normalizer 
    pNorm = pNormalize(
        maxPer =quantiles['high'][str(cfg.dataset.kwargs.timeperiod)],
        minPer =quantiles['low'][str(cfg.dataset.kwargs.timeperiod)]
    )

    # img_transforms define transforms to apply on img only
    # RandomApply Define random augmentations to apply on img with prob p:
   
    img_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
            transforms.ColorJitter()
            ]),p=0.2),
        pNorm
    ])
    
    #imlab_transforms define random flips/rotations to apply on both img and labl with prob p
    img_label_transform = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90)
            ]),p=0.4)
    ])

    return(img_transforms,img_label_transform,pNorm)


def get_dataLoaders(cfg):

    # Get image transforms
    img_transforms,img_label_transforms,pNorm = get_transforms(cfg)

    # parameters for dataset 
    train_kwargs = cfg.dataset.train_kwargs
    val_kwargs = cfg.dataset.val_kwargs
    test_kwargs = cfg.dataset.test_kwargs

    # parameters for dataloaders
    loader_train_kwargs=cfg.data_loader.train_kwargs
    loader_test_kwargs =cfg.data_loader.test_kwargs
    
   
    # Define datasets for training & validation
    training_set = sentinel( **train_kwargs,
                            img_transform=img_transforms,
                            transforms=img_label_transforms
                           )
    
    validation_set = sentinel(**val_kwargs,
                              img_transform=pNorm 
                             )
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, **loader_train_kwargs)
    validation_loader = DataLoader(validation_set, **loader_test_kwargs)

    if cfg.dataset.test_kwargs.data == 'val':
        test_loader = validation_loader
        print('Validation set is used for final testing')
    elif cfg.dataset.test_kwargs.data == 'test':
        test_set= sentinel(**test_kwargs)
        test_loader=DataLoader(test_set, **loader_test_kwargs)
        print('Test set is used for final testing. \nTest set has {} instances'.format(len(test_set)))
    
    # Report split sizes 
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances\n'.format(len(validation_set)))    

    return(training_loader,validation_loader,test_loader)
        
        
def main():
    # get config file
    cfg = get_config()

    # set which class counts to use depending on which set is used for final testing
    test_classCounts=classCounts[cfg.dataset.test_kwargs.data]
    train_classCounts= classCounts['train']
    val_classCounts= classCounts['val']

    nClasses = len(list(set(val['train_id'] for val in class_dict.values()))) + 1
    cfg.config.nClasses = nClasses
    
    # update unique dataset train kwargs with non unique dataset kwargs 
    cfg.dataset.train_kwargs.update(cfg.dataset.kwargs)
    cfg.dataset.test_kwargs.update(cfg.dataset.kwargs)
    cfg.dataset.val_kwargs.update(cfg.dataset.kwargs)
  
    #for testing purpouse 
    dry_run = cfg.config.dry_run 
    
    # train on GPU if no_cuda is false (and cuda is available) 
    use_cuda = not cfg.config.no_cuda and torch.cuda.is_available()
    
    # how often (#Batch) to log results in terminal  
    log_intervall=cfg.config.log_intervall 
    
    # Save model for future inference
    save_model=cfg.config.save_model # Boolean 
      
    # Define savedir
    savedir = get_savedir(cfg) 

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

    #manual seed
    if manual_seed:
        torch.manual_seed(seed)
      
    # if use_cuda => use cuda_kwargs  
    if use_cuda:
        cfg.data_loader.train_kwargs.update(cfg.cuda_kwargs)
        cfg.data_loader.test_kwargs.update(cfg.cuda_kwargs)
    
    
    print(' \n PyTorch 3D-Unet running on device:', device)
    print_cfg(cfg)
    
    #writer is for tensorboard
    train_writer = SummaryWriter(os.path.join(savedir,'train_LOSS'))
    val_writer = SummaryWriter(os.path.join(savedir,'val_LOSS'))
    
    #get data Loaders 
    training_loader,validation_loader,test_loader = get_dataLoaders(cfg)
    
    # batch sample
    img, labl = iter(training_loader).next()

    # Define model
    model = UNET(in_channels=img.shape[1],classes=nClasses).to(device)
    
    if load_model: 
        model.load_state_dict(torch.load(load_path))
    
    # specify optimizer
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    
    
    # Compute Cross Entropy Loss Weights
    if cfg.loss.crossEntropy.weighted:
        ce_weights_train = ce_weights
    else:
        ce_weights_train = None

    
 
    # Specify loss functions, ce = Cross Entropy Loss, ftl = Focal Tversky Loss
    loss_ce = nn.CrossEntropyLoss(weight=ce_weights_train,ignore_index=0).to(device)    
    loss_ftl = focalTverskyLoss(**cfg.loss.focalTversky_kwargs,device=device).to(device)

    # initiate best_vloss/iou
    best_vloss = 1_000_000.
    best_iou = 0

    # for time tracking
    time_est = 0
    tic_start = datetime.now()    
    
    save_cfg(cfg,savedir)

    # train/test loop
    for i in range(epochs):
        epoch = i+1
        
        if not load_model:
            # Train one epoch
            tic = time.perf_counter() 
        avg_loss = train(cfg, model, device, training_loader, optimizer, loss_ce, loss_ftl, epoch, train_writer,train_classCounts)
            
        if np.isnan(avg_loss):
            break
            
        # validate / test
        avg_vloss,iou = test(cfg, model, device, validation_loader, loss_ce,loss_ftl,val_classCounts)
        
        time_epoch = round((time.perf_counter()-tic)/60.0,2)
        time_est = datetime.now()-tic_start
        
        print('Epoch: {}, Time epoch (min): {}, total_time: {}, Loss train: {}, Loss valid: {}, IOU: {}'.format(epoch,
                                                                                                                time_epoch,time_est,
                                                                                                                round(avg_loss,2),
                                                                                                                round(avg_vloss,2),
                                                                                                                round(iou.item(),2))
             )

        # (tensorboard) Log the running loss averaged per batch for both training and validation
        if cfg.config.tensorboard:
            train_writer.add_scalar('LOSS',avg_loss,epoch)
            val_writer.add_scalar('LOSS',avg_vloss,epoch)
            val_writer.add_scalar('IOU',iou.item(),epoch)
            train_writer.flush()
            val_writer.flush()
        
        # track best performance, and save the model's state
        if iou > best_iou:
            best_iou = iou
            if save_model:
                model_path = os.path.join(savedir,'saved_model_IOU/model_epoch_{}.pt'.format(epoch))
                best_model_iou_state_dict = model.state_dict()
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                torch.save(best_model_iou_state_dict, model_path)
                
        # track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        
            # save model for inference
            if save_model:
                model_path = os.path.join(savedir,'saved_model_loss/model_epoch_{}.pt'.format(epoch))
                best_model_state_dict = model.state_dict()
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                torch.save(best_model_state_dict, model_path)
                
    # best model evaluation
    model.load_state_dict(best_model_iou_state_dict)
    model.to(device)
    evals = eval(cfg,model,device,test_loader,test_classCounts)



    


if __name__ == '__main__':
    main()
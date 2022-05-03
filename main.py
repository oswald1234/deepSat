from __future__ import print_function
import argparse
from datetime import datetime
from locale import normalize
from math import degrees
from sklearn.preprocessing import KernelCenterer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.datasets import sentinel
from model.models import UNET

from train import train,test

# for config 
import yaml
#makes dicts nested for nested calls
import munch


#TODO: save cfg

# get config file "config.yaml"
def get_conf(path='config.yaml'):
    with open(path) as file:
        try:
            return(munch.munchify(yaml.safe_load(file)))
        except yaml.YAMLError as exc:
            print(exc)
            
# use this to start with "main.py --config_file config.yaml" to define wich config file to use 
def get_config():
    p = argparse.ArgumentParser(description='Path to config file')
    p.add_argument('--config_file', metavar='PATH', default='config.yaml',
                    help='path to a configuration file')
    arg = p.parse_args()
    
    
    with open(arg.config_file) as yaml_file:
        try:
            cfg = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    return munch.munchify(cfg) 

#TODO: make pretty/cleanup, there is probably some function for this in munch or maybe yaml docs
def print_cfg(cfg):
    print('\nPyTorch 3D-Unet running on device:', cfg.train.device)
    print('\nsave-model:',cfg.config.save_model)
    print('log-interval:',cfg.config.log_intervall)
    print('seed:',cfg.config.manual_seed)
    print('learning rate:',cfg.train.lr)
    print('epochs:', cfg.train.epochs)
    
    print('\ntest_kwargs:')
    for kwarg in cfg.dataset.test.kwargs: print(kwarg,cfg.dataset.test.kwargs[kwarg])
    print('\ntrain_kwargs:')
    for kwarg in cfg.dataset.train.kwargs: print(kwarg,cfg.dataset.train.kwargs[kwarg])
    
    if cfg.config.dry_run:
        print(' \n Dry run! (only for testing!)')
        
def main():
    # get config file
    cfg = get_conf()
    
    # if use_cuda = TRUE,  if cuda (GPU) is available and config no_cuda = false  
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
    
    #img_labl_transforms define random flips/rot to apply on both img and labl with prob p
    imlab_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90)
            ]),p=0.4)
    ]) 

    #TODO: add real std and mean values
    normalize = transforms.Normalize(
            std=[427.1248, 337.7532, 305.9428, 944.7454, 437.4391, 711.1324, 889.7389,959.4146, 727.0137, 625.2451],
            mean=[1,1,1,1,1,1,1,1,1,1]
            )

    # img_transforms define transforms to apply on img only
    # Random Apply Define random augmentations to apply on img with prob p:
    #TODO: see kernel_size and sigma used in kth-exjobb
    img_transforms = transforms.Compose([
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ColorJitter()
            ]),p=0.4),
        normalize
    ])

   

    # Create datasets for training & validation,
    training_set = sentinel(root_dir=cfg.dataset.train.root, img_transform=img_transforms,transforms=imlab_transforms, rgb = cfg.dataset.rgb)
    validation_set = sentinel(root_dir=cfg.dataset.test.root, img_transform=normalize, rgb = cfg.dataset.rgb)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, **cfg.dataset.train.kwargs)
    validation_loader = DataLoader(validation_set, **cfg.dataset.test.kwargs)
    
    # data sample
    img, labl = iter(training_loader).next()
    
    # Report split sizes
    print('\nTraining set has {} instances'.format(len(training_set)))
    print('\nValidation set has {} instances'.format(len(validation_set)))

    model = UNET(in_channels=img.shape[1]).to(cfg.train.device)

    optimizer = optim.NAdam(model.parameters(), lr=cfg.train.lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/test_run_vm_{}'.format(timestamp))

    best_vloss = 1_000_000.

    for epoch in range(1, cfg.train.epochs + 1):

        # Train one epoch
        avg_loss = train(cfg, model, cfg.train.device, training_loader,
                         optimizer, loss_fn, epoch, writer)
        # validate
        avg_vloss = test(model, cfg.train.device, validation_loader, loss_fn)

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
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            
            # only saves models better than previous
            if cfg.config.save_model:
                torch.save(model.state_dict(), model_path)
                
                
        


if __name__ == '__main__':
    main()

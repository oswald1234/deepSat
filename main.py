from __future__ import print_function
import argparse
from datetime import datetime
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


def get_conf(path='config.yaml'):
    with open(path) as file:
        try:
            return(munch.munchify(yaml.safe_load(file)))
        except yaml.YAMLError as exc:
            print(exc)
            
# use this to start with main.py config.yaml
# need to test functionality before use
def get_config(file='config.yaml'):
    p = argparse.ArgumentParser(description='')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    arg = p.parse_args()
    
    
    with open(file) as yaml_file:
        try:
            return(munch.munchify(yaml.safe_load(file)))
        except yaml.YAMLError as exc:
            print(exc)

    return munch.munchify(cfg) 

##TODO: make pretty there is probably some function for this in munch or maybe yaml docs
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
    # get config
    cfg = get_conf()
    
    
    use_cuda = not cfg.train.no_cuda and torch.cuda.is_available()
    cfg.train.device = torch.device('cuda' if use_cuda else 'cpu')
    #manual seed
    if cfg.config.manual_seed:
        torch.manual_seed(cfg.config.seed)
      
  
    
    if use_cuda:
        cfg.dataset.train.kwargs.update(cfg.cuda_kwargs)
        cfg.dataset.test.kwargs.update(cfg.cuda_kwargs)
    
    print_cfg(cfg)
    
    # todo: save cfg
    
    
    # TODO: add augmentation this transform is applied to  both training and validation at the moment
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets for training & validation,
    training_set = sentinel(root_dir=cfg.dataset.train.root, img_transform=transform, rgb = cfg.dataset.rgb)
    validation_set = sentinel(root_dir=cfg.dataset.test.root, img_transform=transform, rgb = cfg.dataset.rgb)

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
            if args.save_model:
                torch.save(model.state_dict(), model_path)
                
                
        


if __name__ == '__main__':
    main()

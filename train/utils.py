# for config 
from distutils.command.config import config
import yaml
import os
#makes dicts nested for nested calls
import munch
import torch
import numpy as np
from tqdm import tqdm
import argparse
from dataset.utils import classCount
from train.metrics import computeConfMats,valMetric


# train one epoch
def train(cfg, model, device, train_loader, optimizer, loss_ce, loss_ftl, epoch, tb_writer):

    running_loss = 0.
    last_loss = 0.
    batch_idx = 0

    model.train(True)
    ninstances = len(train_loader.dataset)
    log_intervall = cfg.config.log_intervall if cfg.config.log_intervall != 0 else 99999
    w = cfg.loss.weight

    for inp, target in train_loader:
        batch_idx += 1
        # Every data instance is an input (X) + target (y) pair
        inp, target = inp.to(device), target.to(device)

        # zero gradients for every batch
        optimizer.zero_grad()

        # make predictions for batch
        output = model(inp)

        
        # compute loss and gradients
        
        loss = w*loss_ce(output, target) + (1-w)*loss_ftl(output, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # number of training imgs
   
        #if cfg.config.dry_run:
        #    dryprint(loss,inp)
      
        # report (epoch loss) every log_intervall 
        if batch_idx % log_intervall == 0:
            last_loss = running_loss/batch_idx
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                epoch,cfg.train.epochs, batch_idx*len(inp), ninstances,
                100.*batch_idx*len(inp) / ninstances, last_loss)
            )
            # Report to tensor board
            #tb_x = (epoch-1) * len(train_loader) + batch_idx    # x value
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x) 
            

            if cfg.config.dry_run:
                break

    return running_loss/batch_idx #epoch mean


def test(cfg, model, device, validation_loader, loss_ce, loss_ftl,val_classCounts):
    model.eval()
    running_vloss = 0.
    batch_idx= 0.
    w= cfg.loss.weight
    cMats=torch.zeros((cfg.config.nClasses-1,2,2),dtype=torch.int32)
    iou=0

    with torch.no_grad():
        for vinputs, vtarget in validation_loader:
            vinputs, vtarget = vinputs.to(device), vtarget.to(device)
            voutputs = model(vinputs)
            
            # validation loss
         
            vloss = w*loss_ce(voutputs, vtarget) + (1-w)*loss_ftl(voutputs, vtarget)
           
            running_vloss += vloss.item()
               
            batch_idx += 1
                
            #if cfg.config.dry_run:
            #    print('V_loss:', round(vloss.item(),2),'shape:',vinputs.shape)
                
                
            # prediction
            pred = torch.nn.functional.softmax(voutputs,dim=1)
            pred = torch.argmax(pred,dim=1)
            cMats += computeConfMats(vtarget.cpu(),pred.cpu())
            
    avg_vloss = running_vloss / batch_idx
    iou = valMetric(cMats,val_classCounts)
    return (avg_vloss,iou) 


    
    
    
    
########### helper functions ############# 

# check if tensor contains inf values
def isinf(tensor):
    return(torch.any(torch.isinf(tensor)).item())

def dryprint(loss,inp):
    maxv=round(inp.max().cpu().numpy().item(),2)
    minv=round(inp.min().cpu().numpy().item(),2)
    loss=round(loss.item(),2)
    print('Loss:', loss, inp.shape, '| isinf:',isinf(inp),'| min-max (all bands):',(minv,maxv))

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

    cfg = get_conf(path=arg.config_file)


    return cfg 

# print config
def print_cfg(cfg):
    #from pprint import pprint
    #pprint(munch.unmunchify(cfg))
    for conf in cfg:
        print('\v',conf)
        for key in cfg[conf]:
            if key.__contains__('kwargs'):
                print('  ',key)
                for kwarg in cfg[conf][key]:
                    print('      ',kwarg,'\t',cfg[conf][key][kwarg])
            else:
                print('    ',key,'\t',cfg[conf][key])
    print('\v\v')

def save_cfg(cfg,savedir):
    with open(os.path.join(savedir,'config.yaml'),'w') as file:
        yaml.dump(config,file)


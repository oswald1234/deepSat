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
from datetime import datetime

from zmq import device
from dataset.utils import classCount
from train.metrics import computeConfMats, valMetric, computeClassMetrics, wma, printClassMetrics, printModelMetrics, plotConfusionMatrices, plotConfusionMatrix,plot_batch,plot_sample

# train one epoch
def train(cfg, model, device, train_loader, optimizer, loss_ce, loss_ftl, epoch, tb_writer, train_classCounts):

    running_loss = 0.
    last_loss = 0.
    batch_idx = 0
    cMats=torch.zeros((cfg.config.nClasses-1,2,2),dtype=torch.int32)
    iou = 0

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

        if w!='None':
            loss = w*loss_ce(output, target) + (1-w)*loss_ftl(output, target)
        else:
            loss= loss_ce(output, target) + loss_ftl(output, target)

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
            if np.isnan(last_loss):
                return(running_loss)
   
            # prediction for IOU 
            #pred = torch.nn.functional.softmax(output,dim=1)
            #pred = torch.argmax(pred,dim=1)
            #cMats += computeConfMats(target.cpu(),pred.cpu())
            #iou = valMetric(cMats,train_classCounts)    
             #Report to tensor board
            tb_x = (epoch-1) * len(train_loader) + batch_idx    # x value
            tb_writer.add_scalar('Loss/train', last_loss, tb_x) 
            #tb_writer.add_scalar('IOU/train', iou, tb_x) 
            

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
            if w!='None':
                vloss = w*loss_ce(voutputs, vtarget) + (1-w)*loss_ftl(voutputs, vtarget)
            else:
                vloss= loss_ce(voutputs, vtarget) + loss_ftl(voutputs, vtarget)
           
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

def eval(cfg,best_model,device,test_loader,test_classCounts):

    best_model.eval()
    cMats=torch.zeros((cfg.config.nClasses-1,2,2),dtype=torch.int32)
    predarr = torch.tensor([],dtype=torch.long,device=device)   
    labelarr = torch.tensor([],dtype=torch.long,device=device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device),labels.to(device)
            outputs = best_model(images)
            preds = torch.nn.functional.softmax(outputs,dim=1)
            preds = torch.argmax(preds,dim=1)
              
            # Flatten dimensions BxHxW --> B*H*W and concatenate
            predarr = torch.cat((predarr, preds.reshape(-1)))
            labelarr = torch.cat(( labelarr, labels.reshape(-1)))
            
            cMats += computeConfMats(labels.cpu(),preds.cpu())
          
    plot_batch(preds,labels,images,path=cfg.config.savedir)
    plot_best_worst(preds,labels,images,path=cfg.config.savedir)
    labels,preds,images = labels.cpu(),preds.cpu(),images.cpu()
    #plot_batch(preds,labels,images,path=cfg.config.savedir)
    #plot_best_worst(preds,labels,images,path=cfg.config.savedir)
    report_metrics(cMats,labelarr.cpu(),predarr.cpu(),test_classCounts,TB=cfg.config.tensorboard,path=cfg.config.savedir)
    
 
########### helper functions ############# 

# to plot best and worst prediction in batch
# input batch of preds labels and images
# finds best and worst prediction and plots them
# uses function plot_sample from train.metrics.py 
def plot_best_worst(preds,labels,images,path):

    #plot sample of best and worst in batch
    sum_eq = torch.sum(torch.eq(preds,labels),dim=[1,2])

    best= torch.where(torch.max(sum_eq)==sum_eq)
    worst = torch.where(torch.min(sum_eq)==sum_eq)

    wlabl=labels[worst].squeeze()
    wpred=preds[worst].squeeze()
    wimg=images[worst].squeeze()

    blabl=labels[best].squeeze()
    bpred=preds[best].squeeze()
    bimg=images[best].squeeze()
    
    plot_sample(wpred,wlabl, wimg[0:3,:,:],fn='batch_worst.png',path=path) 
    plot_sample(bpred,blabl, bimg[0:3,:,:],fn='batch_best.png',path=path)


def report_metrics(cMats,labelarr,predarr,test_classCounts,TB=True,path='runs/'):
     # Compute class and model metrics
    class_metrics = computeClassMetrics(cMats)
    model_metrics = wma(class_metrics,test_classCounts)

    # Print model metrics
    printModelMetrics(model_metrics,path=path)
    # Print model metrics to TENSORBOARD. Default path = 'runs/Model_Metrics'
    printModelMetrics(model_metrics,TB=TB,path=path)
   
    # Print class Metrics
    printClassMetrics(class_metrics,test_classCounts,path=path)
    printClassMetrics(class_metrics,test_classCounts,TB=TB,path=path)
    
    # Plot confusion matrices to TENSORBOARD. Default path = 'runs/Confusion_Matrices'
    plotConfusionMatrices(cMats,TB=TB,path=path)

    # Plot N_CLASS X N_CLASS confusion matrix
    #plotConfusionMatrix(yTrue=labelarr,yPred=predarr,path=path)

    plotConfusionMatrix(yTrue=labelarr,yPred=predarr,TB=TB,path=path)

# check if tensor contains inf values
def isinf(tensor):
    return(torch.any(torch.isinf(tensor)).item())

def dryprint(loss,inp):
    maxv=round(inp.max().cpu().numpy().item(),2)
    minv=round(inp.min().cpu().numpy().item(),2)
    loss=round(loss.item(),2)
    print('Loss:', loss, inp.shape, '| isinf:',isinf(inp),'| min-max (all bands):',(minv,maxv))




############## Config Utils ############    
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
    
    cfg = munch.unmunchify(cfg)
    path= os.path.join(savedir,'config.yaml')
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
            
    with open(path,'w') as file:
        yaml.safe_dump(munch.unmunchify(cfg),file)

def get_savedir(cfg):
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    if cfg.config.dry_run:
        savedir='runs/dry_run_{}'.format(timestamp)
    else: 
        loss = '{WEIGHTED}{CE}{ftl}'.format(WEIGHTED= 'weighted_' if cfg.loss.weight!='None' else '',
                                        CE='WCE' if cfg.loss.crossEntropy.weighted else 'CE',
                                        ftl= '_FTL' if cfg.loss.use_focal_tversky else ''
                                       )
        band='rgb' if cfg.dataset.kwargs.rgb else 'all'
        savedir = 'runs/{loss}_{epochs}_epochs_{band}_bands_TP{timeperiod}_{timestamp}'.format(loss=loss,
                                                                               epochs = cfg.train.epochs,
                                                                               band=band,
                                                                               timeperiod=cfg.dataset.kwargs.timeperiod,
                                                                               timestamp= timestamp
                                                                              )
    cfg.config.savedir=savedir
    return savedir
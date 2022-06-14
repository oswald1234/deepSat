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
from preprocess.classDict import   class_dict
from dataset.utils import pNormalize, crossEntropyLossWeights, classCount
from train.loss import focalTverskyLoss
from model.models import UNET
import numpy as np
import time

from train.utils import train, test, get_conf, get_config, print_cfg,save_cfg,get_savedir, eval

import munch
import optuna
from optuna.trial import TrialState


# timeperiod 1 98% quantiles
q_hi_1 = torch.tensor([2102.0, 1716.0, 1398.0, 4732.0, 2434.42919921875, 3701.759765625, 4519.2177734375, 4857.7734375, 3799.80322265625, 3008.8935546875])
q_lo_1 = torch.tensor([102.0, 159.0, 107.0, 77.0, 106.98081970214844, 79.00384521484375, 86.18966674804688, 70.40167236328125, 50.571197509765625, 36.95356750488281])

#timeperiod 2 98% quantiles
q_hi_2 = [1600.0, 1470.0, 1528.0, 4816.0, 2091.430419921875, 3938.1103515625, 4561.27294921875, 4804.4521484375, 2890.80810546875, 2196.6494140625]
q_lo_2 = [22.0, 41.0, 1.0, 1.0, 27.108013153076172, 4.010444641113281, 4.878728866577148, 3.9264116287231445, 11.01298999786377, 13.7161865234375]

#old with added weight on "other roads and..."
ce_weights = torch.tensor([0.0000e+00, 5.1362e+02, 1.1472e+02, 4.4708e+01, 1.7092e+01, 1.6746e+01,
        4.4391e+01, 1.6548e+01, 1.1023e+02, 1.0783e+02, 1.1555e+02, 1.0042e+03,
        1.6943e+02, 9.5672e+01, 4.4588e+02, 3.8277e+02, 3.2361e+01, 3.2498e+01,
        2.8015e+00, 2.0817e+04, 3.2352e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
        4.9671e+01, 5.8634e+03, 4.5644e+01, 6.7183e+00])

val_classCounts = torch.tensor([  
    70285,    8015,   68954,  181556,  534386,  591861,  211741,  473394,
    91870, 1137165,   66135,     662,  124072,   70080,   15372,   30516,
    262138,  220720, 3414479,     162, 2646013,       0,       0, 9503818,
    163217,    1668,  173970, 1499095], 
    dtype=torch.int32)

train_classCounts= torch.tensor([  
    535419,   138327,   619285,  1589119,  4156781,  4242609, 1600499, 4293367,
    644562,  9073507,   614877,    70747,   419341,  742613,  159342,   185612,
    2195461,  2186212, 25360177,     3413, 21960385, 0,       0,      71047030,
    1430349,    12117,  1556533, 10575180],
    dtype=torch.int32)

test_classCounts=torch.tensor([
    64911,   10252,   69500,  248063,  565968,  585386,  198120,  558537,
    71158, 1161760,   81514,    4988,   57822,   77373,   25242,   19492,
    329747,  244345, 3099608,       0, 2497249,       0,       0, 8873105,
    128908,    1642,  235669, 1433481], 
    dtype=torch.int32)

def get_transforms(cfg):

    # quantiles
    if cfg.dataset.kwargs.timeperiod ==1:
        q_hi = q_hi_1
        q_lo = q_lo_1
    else:
        q_hi = q_hi_2
        q_lo = q_lo_2
    
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
        
def objective(trial):
    # get config file
    cfg = get_config()

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

    # Optuna parameters
    #lr = cfg.optimizer.lr
    lr = trial.suggest_float('lr',1e-5,1e-2)
    alpha = trial.suggest_float('alpha',0.5, 1)
    beta = 1-alpha
    gamma = trial.suggest_float('gamma',1, 3)

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
    loss_ftl = focalTverskyLoss(**cfg.loss.focalTversky_kwargs).to(device)
    loss_ftl = focalTverskyLoss(smooth=1,alpha=alpha,beta=beta,gamma=gamma).to(device)
    
    # initiate best_vloss/iou
    best_vloss = 1_000_000.
    best_iou = 0
    
    # for tracking of time 
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
                                                                                                                time_epoch,
                                                                                                                time_est,
                                                                                                                round(avg_loss,2),
                                                                                                                round(avg_vloss,2),
                                                                                                                round(iou.item(),2)
                                                                                                               )
             )

        # (tensorboard) Log the running loss averaged per batch for both training and validation
        if cfg.config.tensorboard:
            train_writer.add_scalar('LOSS',avg_loss,epoch)
            val_writer.add_scalar('LOSS',avg_vloss,epoch)
            val_writer.add_scalar('IOU',iou.item(),epoch)
            train_writer.flush()
            val_writer.flush()
        
        
        trial.report(iou,epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        '''
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
    '''
    
    return iou

    

if __name__ == '__main__':
    study  = optuna.create_study(direction='maximize',study_name="testrun",pruner=optuna.pruners.HyperbandPruner(),load_if_exists=True)
    study.enqueue_trial(
        {
            "lr": 1e-4,
            "alpha": 0.7,
            "gamma": 1.33
        }
        )
    study.optimize(objective,n_trials=40  )
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print(' Number of finished trials: ', len(study.trials))
    print(' Number of pruned trials: ', len(pruned_trials))
    print(' Number of complete trials: ', len(complete_trials))

    print(' Best trial:')
    trial = study.best_trial

    print(' Value: ', trial.value)

    print(' Params ')
    for key, val in trial.params.items():
        print('    {}: {}'.format(key,val))
        

        
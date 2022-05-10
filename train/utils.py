# for config 
import yaml
#makes dicts nested for nested calls
import munch
import torch
import numpy as np


def isinf(tensor):
    return(torch.any(torch.isinf(tensor)).item())

    
# train one epoch
def train(cfg, model, device, train_loader, optimizer, loss_fn, epoch, tb_writer):

    running_loss = 0.
    last_loss = 0.
    ninstances =0
    model.train(True)

    for idx, (inp, target) in enumerate(train_loader):
        batch_idx = idx + 1
        # Every data instance is an input (X) + target (y) pair
        inp, target = inp.to(device), target.to(device)

        # zero gradients for every batch
        optimizer.zero_grad()

        # make predictions for batch
        output = model(inp)
        
        # compute loss and gradients
        loss = loss_fn(output, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # number of training imgs
        ninstances += 1
        if cfg.config.dry_run:
            print('Loss:', round(loss.item(),2),inp.shape,'| isinf:',isinf(inp),'| min-max (all bands):' ,(inp.min().cpu().numpy().item(),inp.max().cpu().numpy().item()))

        # report (epoch loss) every log_intervall 
        if batch_idx % cfg.config.log_intervall == 0:
            last_loss = running_loss/ninstances
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx*len(inp), len(train_loader.iterable.dataset),
            #    100.*batch_idx*len(inp) / len(train_loader.iterable.dataset), last_loss)
            #)
            # Report to tensor board
            tb_x = (epoch-1) * len(train_loader) + ninstances    # x value
            tb_writer.add_scalar('Loss/train', last_loss, tb_x) 
            

            #if cfg.config.dry_run:
                #break

    return running_loss/ninstances #epoch mean


def test(cfg,model, device, validation_loader, loss_fn):
    model.eval()
    running_vloss = 0.
    ninstance= 0.
    
    with torch.no_grad():
        for vinputs, vtarget in validation_loader:
            vinputs, vtarget = vinputs.to(device), vtarget.to(device)
            voutputs = model(vinputs)
            
            # validation loss
            vloss = loss_fn(voutputs, vtarget)
            if not np.isnan(vloss.item()):
                running_vloss += vloss.item()
                #number of instances
                ninstance += 1
                
            if cfg.config.dry_run:
                print('V_loss:', round(vloss.item(),2),'shape:',vinputs.shape)
                
                
            # prediction
            #pred = torch.nn.functional.softmax(voutputs,dim=1)
            #pred = torch.argmax(pred,dim=1)
            
    avg_vloss = running_vloss / ninstance

    return avg_vloss


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
    if cfg.config.manual_seed:
        print('seed:',cfg.config.seed)
    print('learning rate:',cfg.train.lr)
    print('epochs:', cfg.train.epochs)
    
    print('\ntest_kwargs:')
    for kwarg in cfg.dataset.test.kwargs: print(kwarg,cfg.dataset.test.kwargs[kwarg])
    print('\ntrain_kwargs:')
    for kwarg in cfg.dataset.train.kwargs: print(kwarg,cfg.dataset.train.kwargs[kwarg])
    
    if cfg.config.dry_run:
        print(' \n Dry run! (only for testing!)')
import torch

# train one epoch
def train(cfg, model, device, train_loader, optimizer, loss_fn, epoch, tb_writer):

    running_loss = 0.
    last_loss = 0.
    ninstances =0
    model.train(True)

    for idx, (input, target) in enumerate(train_loader):
        batch_idx = idx + 1
        # Every data instance is an input (X) + target (y) pair
        input, target = input.to(device), target.to(device)

        # zero gradients for every batch
        optimizer.zero_grad()

        # make predictions for batch
        output = model(input)

        # compute loss and gradients
        loss = loss_fn(output, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # number of training imgs
        ninstances += input.shape[0]

        # report (epoch loss) every log_intervall 
        if batch_idx % cfg.config.log_intervall == 0:
            last_loss = running_loss/ninstances
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(input), len(train_loader.dataset),
                100.*batch_idx / len(train_loader), last_loss)
            )
            # Report to tensor board
            tb_x = (epoch-1) * len(train_loader) + ninstances    # x value
            tb_writer.add_scalar('Loss/train', last_loss, tb_x) 
            

            if cfg.config.dry_run:
                break

    return running_loss/ninstances #epoch mean


def test(model, device, validation_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    
    with torch.no_grad():
        for vinputs, vtarget in validation_loader:
            vinputs, vtarget = vinputs.to(device), vtarget.to(device)
            voutputs = model(vinputs)
            
            # validation loss
            vloss = loss_fn(voutputs, vtarget)
            running_vloss += vloss

            # prediction
            pred = torch.nn.functional.softmax(voutputs,dim=1)
            pred = torch.argmax(pred,dim=1)


    avg_vloss = running_vloss / (len(validation_loader.dataset))
    return avg_vloss
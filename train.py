import torch


def train(args, model, device, train_loader, optimizer, loss_fn, epoch, tb_writer):

    running_loss = 0.
    last_loss = 0.
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
        if batch_idx % args.log_interval == 0:
            last_loss = running_loss/args.log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(input), len(train_loader.dataset),
                100.*batch_idx / len(train_loader), last_loss)
            )
            # Report to tensor board
            tb_x = (epoch-1) * len(train_loader) + batch_idx*len(input)
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

            if args.dry_run:
                break

    return last_loss


def test(model, device, validation_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    
    with torch.no_grad():
        for vinputs, vtarget in validation_loader:
            vinputs, vtarget = vinputs.to(device), vtarget.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vtarget)
            running_vloss += vloss

    avg_vloss = running_vloss / (len(validation_loader.dataset))
    return avg_vloss
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

## root to train and validation set
root = 'processed-data/TCI_256_split/Tidsperiod-1/train'
rootv = 'processed-data/TCI_256_split/Tidsperiod-1/val'


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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 3D-unet')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4,
                        metavar='N', help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true',
                        default=True, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducability (default: 1)', metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true', default=True)
    args = parser.parse_args()

    # use cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # seed for reproducability
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')
    print('running on device:', device)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dry_run:
        print('Dry run! (for testing)')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets for training & validation,
    training_set = sentinel(root_dir=root, img_transform=transform)
    validation_set = sentinel(root_dir=rootv, img_transform=transform)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, **train_kwargs)
    validation_loader = DataLoader(validation_set, **test_kwargs)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    model = UNET().to(device)

    optimizer = optim.NAdam(model.parameters(), lr=args.lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/s2_trainer_{}'.format(timestamp))

    best_vloss = 1_000_000.

    for epoch in range(1, args.epochs + 1):

        # Train one epoch
        avg_loss = train(args, model, device, training_loader,
                         optimizer, loss_fn, epoch, writer)
        # validate
        avg_vloss = test(model, device, validation_loader, loss_fn)

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

            if args.save_model:
                torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()

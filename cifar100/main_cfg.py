from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import models
from utils import get_train_valid_loader, get_test_loader
from train import train, get_optim_set, test, valid, save_checkpoint

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')

parser.add_argument('--valid', action='store_true', default=False,
                    help='enable validation')
parser.add_argument('--cfg', default=0, type=int,
                    help='config of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.valid:
    valid_size = 0.1
else:
    valid_size = 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader, valid_loader = get_train_valid_loader(
    data_dir='./data.cifar100',
    batch_size=args.batch_size,
    augment=True,
    random_seed=args.seed,
    valid_size=valid_size,
    shuffle=True,
    show_sample=False,
    **kwargs
)
test_loader = get_test_loader(
    data_dir='./data.cifar100',
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)
 
cfgs=[[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 208, 208, 208],
      [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 1600]] 
cfg = cfgs[args.cfg]

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        cfg = checkpoint['cfg']
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit()

print(cfg)

model = models.__dict__[args.arch](dataset=args.dataset, cfg=cfg)
device = torch.device("cuda" if args.cuda else "cpu")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    avg_loss, train_acc = train(
        model, 
        optimizer, 
        epoch=epoch, 
        device=device,
        train_loader=train_loader,
        valid=args.valid, 
        valid_size=valid_size, 
        log_interval=args.log_interval)
    if args.valid:
        prec1 = valid(model, device, valid_loader, valid_size=valid_size)
        test(model, device, test_loader)
    else:
        prec1 = test(model, device, test_loader)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)

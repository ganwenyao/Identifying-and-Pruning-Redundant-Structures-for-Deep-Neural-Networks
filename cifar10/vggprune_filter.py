import argparse
import numpy as np
import os
import shutil
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from utils import get_train_valid_loader, get_test_loader
from prune import prune_filter, prune_filter_l1, prune_layer
from train import train, get_optim_set, test, valid, save_checkpoint

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')

parser.add_argument('--pruneT', type=float, default=0.92, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--freeze-conv', action='store_true', default=False,
                    help='Freeze convolutional layers which are not affected by pruning')
parser.add_argument('--freeze-linear', action='store_true', default=False,
                    help='Freeze fully-connection layers which are not affected by pruning')  
parser.add_argument('--min-filters', default=8, type=int,
                    help='the minimum number of conv layers to be preserve, used to protect the model')

parser.add_argument('--valid', action='store_true', default=False,
                    help='enable validation')

parser.add_argument('--dropT', type=float, default=0.006, metavar='LR',
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.valid:
    valid_size = 0.1
else:
    valid_size = 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader, valid_loader = get_train_valid_loader(
    data_dir='./data.cifar10',
    batch_size=args.batch_size,
    augment=True,
    random_seed=args.seed,
    valid_size=valid_size,
    shuffle=True,
    show_sample=False,
    **kwargs
)
test_loader = get_test_loader(
    data_dir='./data.cifar10',
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

last_prec1 = 0
model = None
cfg = None
if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        cfg = checkpoint['cfg']
        model = vgg(dataset=args.dataset, depth=args.depth, cfg=cfg)
        model.load_state_dict(checkpoint['state_dict'])
        last_prec1 = checkpoint['best_prec1']
        print("=> loaded checkpoint '{}'".format(args.model))
        print(cfg)
        print('best_prec1: ', last_prec1)
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        exit()
        
device = torch.device("cuda" if args.cuda else "cpu")
model = model.to(device)

prec_list = []
prec_list.append(last_prec1)
success_flag_list = []


conv_in_cfg_list = []
for i, c in enumerate(cfg):
    if c != 'M':
        conv_in_cfg_list.append(i)
conv_id = len(conv_in_cfg_list) - 1
conv_id_end = len(conv_in_cfg_list) - 1
while conv_id >= 0:
    while True:
        old_cfg = cfg.copy()
        preserve = []
        prune = []
        linear = []
        print(conv_id)
        cfg_id = conv_in_cfg_list[conv_id]
        if cfg[cfg_id] <= args.min_filters:
            print("obtain minimum channels: %d" % args.min_filters)
            break
        cfg[cfg_id] = int(cfg[cfg_id]/2)
        newmodel = vgg(dataset=args.dataset, cfg=cfg)
        newmodel = newmodel.to(device)
        preserve, prune, linear = prune_filter(
            newmodel, 
            conv_id, 
            conv_id_end, 
            model, 
            args.freeze_conv, 
            args.freeze_linear
        )
        optim_set = get_optim_set(preserve, prune, linear, preserve_lr=0.001, prune_lr=args.lr, linear_lr=0.001)
        optimizer = optim.SGD(optim_set, momentum=args.momentum)
        for epoch in range(args.start_epoch, args.epochs):
            avg_loss, train_acc = train(
                newmodel, 
                optimizer, 
                epoch=epoch, 
                device=device,
                train_loader=train_loader,
                valid=args.valid, 
                valid_size=valid_size, 
                log_interval=args.log_interval
            )
            # test(newmodel, device, test_loader)
            if args.valid:
                prec = valid(newmodel, device, valid_loader, valid_size=valid_size)
            else:
                prec = train_acc
            if prec > args.pruneT:
                break
                
        prec_list.append(prec)
        print('(last_prec1 - prec): %.3f' % (last_prec1 - prec))
        # print('dropT', args.dropT)
        if prec > args.pruneT and (last_prec1 - prec) <= args.dropT:
            last_prec1 = prec
            success_flag_list.append(1)
            del model
            model = newmodel
            print('\nSuccess to prune conv_%d:' % (conv_id+1), cfg, '\n')
        else:
            success_flag_list.append(0)
            cfg = old_cfg
            print('Fail to prune conv_%d:' % (conv_id+1), 'back to:', old_cfg)
            del newmodel
            break
    conv_id -= 1

print('Pruned cfg:')
print(cfg)

torch.save({
    'cfg': cfg,
    'epoch': 0,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_prec1': last_prec1
}, os.path.join(args.save, 'checkpoint.pth.tar.end'))

prec_list_path = os.path.join(args.save, "prec_list.pkl")
with open(prec_list_path, 'wb') as f:
    pickle.dump(prec_list, f)

success_flag_listpath = os.path.join(args.save, "success_flag_list.pkl")
with open(success_flag_listpath, 'wb') as f:
    pickle.dump(success_flag_list, f)

print(prec_list)
print(success_flag_list)
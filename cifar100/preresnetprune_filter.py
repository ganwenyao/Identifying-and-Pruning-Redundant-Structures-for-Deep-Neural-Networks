import argparse
import numpy as np
import os
import shutil
import time
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *
from utils import get_train_valid_loader, get_test_loader
from prune import prune_residual_filter
from train import train, get_optim_set, test, valid, save_checkpoint

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
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
parser.add_argument('--depth', default=56, type=int,
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
parser.add_argument('--reverse', action='store_true', default=False,
                    help='')
parser.add_argument('--dropT', type=float, default=0.006, metavar='LR',
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

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

last_prec1 = 0
model = None
cfg = None
if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        cfg = checkpoint['cfg']
        model = preresnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
        # print(cfg)
        # print(model)
        # print(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        last_prec1 = checkpoint['best_prec1']
        print("=> loaded checkpoint '{}'".format(args.model))
        print(cfg)
        print('best_prec1: ', last_prec1)
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        exit()

# exit()
device = torch.device("cuda" if args.cuda else "cpu")
model = model.to(device)

prec_list = []
prec_list.append(last_prec1)
success_flag_list = []

# Flag is used to identity if the first block 
# which has a projection shortcut has been pruned
n1, flag1 = cfg[0]
n2, flag2= cfg[n1+1]
n3, flag3 = cfg[n1+n2+2]
# cfg1 = cfg[1:n1+1]
# cfg2 = cfg[n1+2:n1+n2+2]
# cfg3 = cfg[n1+n2+3:]

block_id = 0
block_id_end = n1 + n2 + n3 - 1

range_list = list(range(n1 + n2 + n3))
if args.reverse:
    range_list.reverse()
for block_id in range_list:
    prune_success = True
    while prune_success:
        prune_success = False
        for conv_id in [1, 0]:
            block_id_in_cfg = 0
            if block_id < n1:
                block_id_in_cfg = block_id + 1
            elif block_id < n1 + n2:
                block_id_in_cfg = block_id + 2
            else:
                block_id_in_cfg = block_id + 3
            if cfg[block_id_in_cfg][conv_id] <= args.min_filters:
                print("obtain minimum channels: %d" % args.min_filters)
                break
            pruned_channels = int(cfg[block_id_in_cfg][conv_id] / 2)
            old_cfg = copy.deepcopy(cfg)
            cfg[block_id_in_cfg][conv_id] = int(cfg[block_id_in_cfg][conv_id] / 2)
            newmodel = preresnet(dataset=args.dataset, cfg=cfg)
            newmodel = newmodel.to(device)
            preserve, prune, linear = prune_residual_filter(newmodel, block_id, conv_id, model, args.freeze_conv, args.freeze_linear)
            optim_set = get_optim_set(
                preserve, 
                prune, 
                linear, 
                preserve_lr=0.001, 
                prune_lr=args.lr, 
                linear_lr=0.001
            )
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
            drop_threshold = args.dropT
            print('(last_prec1 - prec): %.3f' % (last_prec1 - prec))
            if prec > args.pruneT and (last_prec1 - prec) <= drop_threshold:
                last_prec1 = prec
                success_flag_list.append(1)
                del model
                model = newmodel
                prune_success = True
                print('\nSuccess to prune block_%d_%d:' % (block_id+1,conv_id+1), cfg, '\n')
            else:
                success_flag_list.append(0)
                del newmodel
                print('Fail to prune block_%d_%d:' % (block_id+1,conv_id+1), 'back to:', old_cfg)
                cfg = old_cfg

torch.save({
    'cfg': cfg,
    'epoch': 0,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, os.path.join(args.save, 'checkpoint.pth.tar.end'))

print('Pruned cfg:')
print(cfg)
test(model, device, test_loader)
if args.valid:
    valid(model, device, valid_loader, valid_size=valid_size)


prec_list_path = os.path.join(args.save, "prec_list.pkl")
with open(prec_list_path, 'wb') as f:
    pickle.dump(prec_list, f)

success_flag_listpath = os.path.join(args.save, "success_flag_list.pkl")
with open(success_flag_listpath, 'wb') as f:
    pickle.dump(success_flag_list, f)


print(prec_list)
print(success_flag_list)
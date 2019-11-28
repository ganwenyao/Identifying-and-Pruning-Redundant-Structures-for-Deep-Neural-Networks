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
from prune import prune_block_no_init
from train import train, get_optim_set, test, valid, save_checkpoint

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='svhn',
                    help='training dataset (default: svhn)')
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
parser.add_argument('--valid', action='store_true', default=False,
                    help='enable validation')
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
    valid_len = 60000
else:
    valid_len = 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader, valid_loader = get_train_valid_loader(
    data_dir='./data.svhn',
    batch_size=args.batch_size,
    augment=True,
    random_seed=args.seed,
    valid_len=valid_len,
    shuffle=True,
    show_sample=False,
    **kwargs
)
test_loader = get_test_loader(
    data_dir='./data.svhn',
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
        model.load_state_dict(checkpoint['state_dict'])
        last_prec1 = checkpoint['best_prec1']
        print("=> loaded checkpoint '{}'".format(args.model))
        print(cfg)
        print('best_prec1: ', last_prec1)
        best_prec1 = last_prec1
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

# block_id = 0
block_id_end = n1 + n2 + n3 - 1
block_id = block_id_end
while block_id >= 0:
    n1, flag1 = cfg[0]
    n2, flag2= cfg[n1+1]
    n3, flag3 = cfg[n1+n2+2]
    if block_id in [0, n1, n1+n2]:
        block_id -= 1
        continue
    old_cfg = copy.deepcopy(cfg)
    if block_id < n1:
        cfg.pop(block_id + 1)
        cfg[0][0] -= 1
        if block_id == 0:
            cfg[0][1] = False
    elif block_id < n1 + n2:
        cfg.pop(block_id + 2)
        cfg[n1+1][0] -= 1
        if block_id == n1:
            cfg[n1+1][1] = False
    else:
        cfg.pop(block_id + 3)
        cfg[n1+n2+2][0] -= 1
        if block_id == n1 + n2:
            cfg[n1+n2+2][1] = False

    newmodel = preresnet(dataset=args.dataset, cfg=cfg)
    newmodel = newmodel.to(device)
    prune_block_no_init(newmodel, block_id, model)
    optimizer = optim.SGD(newmodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        avg_loss, train_acc = train(
            newmodel, 
            optimizer, 
            epoch=epoch, 
            device=device,
            train_loader=train_loader,
            valid=args.valid, 
            valid_len=valid_len, 
            log_interval=args.log_interval
        )
        if args.valid:
            prec = valid(newmodel, device, valid_loader, valid_len=valid_len)
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
        print('\nSuccess to prune block_%d:' % (block_id+1), cfg, '\n')
        best_prec1 = prec
    else:
        success_flag_list.append(0)
        del newmodel
        print('Fail to prune block_%d:' % (block_id+1), 'back to:', old_cfg)
        cfg = old_cfg
    block_id -= 1

torch.save({
    'cfg': cfg,
    'epoch': 0,
    'best_prec1': last_prec1,
    'state_dict': model.state_dict(),
}, os.path.join(args.save, 'checkpoint.pth.tar.end'))

print('Pruned cfg:')
print(cfg)

if args.valid:
    valid(model, device, valid_loader, valid_len=valid_len)

prec_list_path = os.path.join(args.save, "prec_list.pkl")
with open(prec_list_path, 'wb') as f:
    pickle.dump(prec_list, f)

success_flag_listpath = os.path.join(args.save, "success_flag_list.pkl")
with open(success_flag_listpath, 'wb') as f:
    pickle.dump(success_flag_list, f)
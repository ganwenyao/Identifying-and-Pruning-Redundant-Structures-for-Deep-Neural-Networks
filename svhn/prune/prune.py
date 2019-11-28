import argparse
import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
from models.preresnet import Bottleneck

def prune_filter(newmodel, conv_id, conv_end_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    conv_id_mod = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_id_mod != conv_id and conv_id_mod != conv_id+1:
                m1.weight.data = m0.weight.data.clone()
                m1.weight.requires_grad = not freeze_conv
                preserve.append(m1)
            else:   
                print('prune conv_%d:' % (conv_id_mod+1), m1)
                prune.append(m1)
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_id_mod != conv_id and conv_id_mod != conv_id+1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                m1.weight.requires_grad = not freeze_conv
                m1.bias.requires_grad = not freeze_conv
                preserve.append(m1)
            else:
                print('prune BN:', m1)
                prune.append(m1)
            conv_id_mod += 1
        elif isinstance(m0, nn.Linear):
            # For the last conv layer, train it and the next fully-connected layer from scratch
            if conv_id < conv_end_id:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.weight.requires_grad = not freeze_linear
                m1.bias.requires_grad = not freeze_linear
                linear.append(m1)
            else:
                print('prune linear:', m1)
                prune.append(m1)
    return preserve, prune, linear

def prune_layer(newmodel, conv_id, conv_end_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    model_list = list(model.modules())
    BN_list = []
    CONV_list = []
    linear_list = []
    for m0 in model.modules():
        if isinstance(m0, nn.Conv2d):
            CONV_list.append(m0)
        elif isinstance(m0, nn.BatchNorm2d):
            BN_list.append(m0)
        elif isinstance(m0, nn.Linear):
            linear_list.append(m0)
    BN_list.pop(conv_id)
    CONV_list.pop(conv_id)
    conv_id_mod = 0
    for m1 in newmodel.modules():
        if isinstance(m1, nn.Conv2d):
            m0 = CONV_list.pop(0)
            if conv_id_mod != conv_id:
                conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
            else:
                print('inti conv_%d:' % (conv_id_mod+1), m1)
                prune.append(m1)
        elif isinstance(m1, nn.BatchNorm2d):
            m0 = BN_list.pop(0)
            if conv_id_mod != conv_id:
                BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
            else:
                print('inti BN:', m1)
                prune.append(m1)
            conv_id_mod += 1
        elif isinstance(m1, nn.Linear):
            m0 = linear_list.pop(0)
            # For the last conv layer(conv_id=conv_end_id+1) 
            # or the last conv layer but one(conv_id=conv_end), 
            # remove it and train the next fully-connected layer from scratch
            if conv_id < conv_end_id:
                linear_trans(m0, m1, freeze=freeze_linear, bias=True, mod_list=linear)
            else:
                print('inti linear:', m1)
                prune.append(m1)
    return preserve, prune, linear

def init_one_layer(newmodel, conv_id, conv_end_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    conv_id_mod = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_id_mod != conv_id:
                m1.weight.data = m0.weight.data.clone()
                m1.weight.requires_grad = not freeze_conv
                preserve.append(m1)
            else:   
                prune.append(m1)
                print('inti conv_%d:' % (conv_id_mod+1), m1)
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_id_mod != conv_id:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                m1.weight.requires_grad = not freeze_conv
                m1.bias.requires_grad = not freeze_conv
                preserve.append(m1)
            else:
                prune.append(m1)
                print('inti BN:', m1)
            conv_id_mod += 1
        elif isinstance(m0, nn.Linear):
            # id=-1 for initializing linear layer
            if conv_id != -1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.weight.requires_grad = not freeze_linear
                m1.bias.requires_grad = not freeze_linear
                linear.append(m1)
            else:
                prune.append(m1)
                print('inti linear:', m1)
    return preserve, prune, linear

def prune_block_no_init(newmodel, block_id, model):
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            conv_trans(conv_first, m1)
        elif isinstance(m1, nn.BatchNorm2d):
            BN_trans(BN_last, m1)
        elif isinstance(m1, nn.Linear):
            linear_trans(linear_last, m1)

    # copy downsample
    for name0, m0 in model.named_modules():
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    for name1, m1 in newmodel.named_modules(): 
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            conv_trans(m0, m1)
            assert(m1.bias == None)

    block_list = []
    for m0 in model.modules():
        if isinstance(m0, Bottleneck):
            block_list.append(m0)
    block_list.pop(block_id)

    for block1 in newmodel.modules():
        if isinstance(block1, Bottleneck):
            block0 = block_list.pop(0)
            for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                if isinstance(m1, nn.Conv2d):
                    if name0 != 'downsample':
                        conv_trans(m0, m1)
                elif isinstance(m1, nn.BatchNorm2d):
                    BN_trans(m0, m1)

def prune_block(newmodel, block_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            conv_trans(conv_first, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
        elif isinstance(m1, nn.BatchNorm2d):
            BN_trans(BN_last, m1, freeze=freeze_conv, mod_list=preserve)

    # copy downsample
    for name0, m0 in model.named_modules():
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    for name1, m1 in newmodel.named_modules(): 
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
            assert(m1.bias == None)

    block_list = []
    for m0 in model.modules():
        if isinstance(m0, Bottleneck):
            block_list.append(m0)
    block_list.pop(block_id)

    block_id_mod = 0
    for block1 in newmodel.modules():
        if isinstance(block1, Bottleneck):
            block0 = block_list.pop(0)
            if block_id_mod != block_id: 
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                    elif isinstance(m1, nn.BatchNorm2d):
                        BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
            else:
                print('prune block_%d:' % (block_id+1), block1)
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            prune.append(m1)
                    elif isinstance(m1, nn.BatchNorm2d):
                        prune.append(m1)
            block_id_mod += 1
    # print(block_id_mod,block_id)
    for m1 in newmodel.modules():        
        if isinstance(m1, nn.Linear):
            if block_id_mod > block_id:
                linear_trans(linear_last, m1, freeze=freeze_linear, bias=True, mod_list=linear)
            else:
                print('init linear:', m1)
                prune.append(m1)
    return preserve, prune, linear

def prune_residual_filter(newmodel, block_id, conv_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            conv_trans(conv_first, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
        elif isinstance(m1, nn.BatchNorm2d):
            BN_trans(BN_last, m1, freeze=freeze_conv, mod_list=preserve)
        elif isinstance(m1, nn.Linear):
            linear_trans(linear_last, m1, freeze=freeze_linear, bias=True, mod_list=linear)

    # copy downsample
    for name0, m0 in model.named_modules():
        # print(name0)
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    for name1, m1 in newmodel.named_modules(): 
        # print(name1)
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
            assert(m1.bias == None)

    block_id_mod = 0
    for block0, block1 in zip(model.modules(), newmodel.modules()):
        if isinstance(block0, Bottleneck):
            if block_id_mod == block_id:
                # print('block_id_mod%d' % block_id_mod, block0)
                conv_id_mod = 0
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m0, nn.Conv2d):
                        if name0 != 'downsample':
                            if conv_id_mod != conv_id and conv_id_mod != conv_id+1:
                                conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                            else:
                                print('prune conv_%d_%d:' % (block_id_mod+1, conv_id_mod+1), m1)
                                prune.append(m1)
                            conv_id_mod += 1
                    elif isinstance(m0, nn.BatchNorm2d):
                        if conv_id_mod != conv_id and conv_id_mod != conv_id+1:
                            BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
                        else:
                            print('prune BN:', m1)
                            prune.append(m1)
            else: 
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                    elif isinstance(m1, nn.BatchNorm2d):
                        BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
            block_id_mod += 1
    return preserve, prune, linear

def init_one_block(newmodel, block_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            m1.weight.data = conv_first.weight.data.clone()

            m1.weight.requires_grad = not freeze_conv
            preserve.append(m1)
        elif isinstance(m1, nn.BatchNorm2d):
            m1.weight.data = BN_last.weight.data.clone()           
            m1.bias.data = BN_last.bias.data.clone()
            m1.running_mean = BN_last.running_mean.clone()
            m1.running_var = BN_last.running_var.clone()

            m1.weight.requires_grad = not freeze_conv
            m1.bias.requires_grad = not freeze_conv
            preserve.append(m1)
        elif isinstance(m1, nn.Linear):
            m1.weight.data = linear_last.weight.data.clone()
            m1.bias.data = linear_last.bias.data.clone()

            m1.weight.requires_grad = not freeze_linear
            m1.bias.requires_grad = not freeze_linear
            linear.append(m1)

    # copy downsample
    for name0, m0 in model.named_modules():
        # print(name0)
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    for name1, m1 in newmodel.named_modules(): 
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            m1.weight.data = m0.weight.data.clone()
            
            m1.weight.requires_grad = not freeze_conv
            preserve.append(m1)

    block_id_mod = 0
    for block0, block1 in zip(model.modules(), newmodel.modules()):
        if isinstance(block0, Bottleneck):
            if block_id_mod == block_id:
                print('inti block_%d:' % (block_id_mod+1), block1)
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            print('inti conv:', m1)
                            prune.append(m1)
                    elif isinstance(m1, nn.BatchNorm2d):
                        print('inti BN:', m1)
                        prune.append(m1)       
            else:
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            m1.weight.data = m0.weight.data.clone()

                            m1.weight.requires_grad = not freeze_conv
                            preserve.append(m1)
                    elif isinstance(m1, nn.BatchNorm2d):
                        m1.weight.data = m0.weight.data.clone()           
                        m1.bias.data = m0.bias.data.clone()
                        m1.running_mean = m0.running_mean.clone()
                        m1.running_var = m0.running_var.clone()

                        m1.weight.requires_grad = not freeze_conv
                        m1.bias.requires_grad = not freeze_conv
                        preserve.append(m1)
            block_id_mod += 1

    return preserve, prune, linear


def init_one_conv_in_block(newmodel, block_id, conv_id, model, freeze_conv, freeze_linear, special_id=-1):
    r"""special_id = 0, prune first conv
        special_id = 1, prune first downsample
        special_id = 2, prune second downsample
        special_id = 3, prune third downsample
    """
    preserve = []
    prune = []
    linear = []
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            if special_id == 0:
                print('inti conv_first:', m1)
                prune.append(m1)
            else:
                conv_trans(conv_first, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
        elif isinstance(m1, nn.BatchNorm2d):
            BN_trans(BN_last, m1, freeze=freeze_conv, mod_list=preserve)
        elif isinstance(m1, nn.Linear):
            linear_trans(linear_last, m1, freeze=freeze_linear, bias=True, mod_list=linear)

    # copy downsample
    for name0, m0 in model.named_modules():
        # print(name0)
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    downsample_id = 0
    for name1, m1 in newmodel.named_modules(): 
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            if downsample_id == special_id - 1:
                print('inti downsample_%d:' % downsample_id, m1)
                prune.append(m1)
            else:
                conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
            downsample_id += 1

    if special_id == -1:
        block_id_mod = 0
        for block0, block1 in zip(model.modules(), newmodel.modules()):
            if isinstance(block0, Bottleneck):
                if block_id_mod == block_id:
                    conv_id_mod = 0
                    for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                        if isinstance(m1, nn.Conv2d):
                            if name0 != 'downsample':
                                if conv_id_mod == conv_id:
                                    print('inti conv_%d_%d:' % (block_id_mod+1, conv_id_mod+1), m1)
                                    prune.append(m1)
                                else:
                                    conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                                conv_id_mod += 1
                        elif isinstance(m1, nn.BatchNorm2d):
                            if conv_id_mod == conv_id + 1:
                                print('inti BN:', m1)
                                prune.append(m1)
                            else:
                                BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
                else: 
                    for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                        if isinstance(m1, nn.Conv2d):
                            if name0 != 'downsample':
                                conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                        elif isinstance(m1, nn.BatchNorm2d):
                            BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
                block_id_mod += 1
    else:
        for block0, block1 in zip(model.modules(), newmodel.modules()):
            if isinstance(block0, Bottleneck):
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            conv_trans(m0, m1, freeze=freeze_conv, bias=False, mod_list=preserve)
                    elif isinstance(m1, nn.BatchNorm2d):
                        BN_trans(m0, m1, freeze=freeze_conv, mod_list=preserve)
    return preserve, prune, linear

def BN_trans(m0, m1, freeze=False, mod_list=None):
    m1.weight.data = m0.weight.data.clone()
    m1.bias.data = m0.bias.data.clone()
    m1.running_mean = m0.running_mean.clone()
    m1.running_var = m0.running_var.clone()
    m1.weight.requires_grad = not freeze
    m1.bias.requires_grad = not freeze
    if mod_list is not None:
        mod_list.append(m1)

def conv_trans(m0, m1, freeze=False, bias=False, mod_list=None):
    m1.weight.data = m0.weight.data.clone()
    m1.weight.requires_grad = not freeze
    if bias:
        m1.bias.data = m0.bias.data.clone()
        m1.bias.requires_grad = not freeze
    if mod_list is not None:
        mod_list.append(m1)

def linear_trans(m0, m1, freeze=False, bias=True, mod_list=None):
    m1.weight.data = m0.weight.data.clone()
    m1.weight.requires_grad = not freeze
    if bias:
        m1.bias.data = m0.bias.data.clone()
        m1.bias.requires_grad = not freeze
    if mod_list is not None:
        mod_list.append(m1)

def prune_residual_l1(newmodel, block_id, conv_id, model, freeze_conv, freeze_linear):
    # copy first conv, last BN and linear_last
    conv_first = None
    BN_last = None
    linear_last = None
    downsample_list = []

    for name0, m0 in model.named_children():
        if isinstance(m0, nn.Conv2d):
            if 'downsample' in name0:
                print('downsample in model')
            else:
                conv_first = m0
        elif isinstance(m0, nn.BatchNorm2d):
            BN_last = m0
        elif isinstance(m0, nn.Linear):
            linear_last = m0

    for m1 in newmodel.children():
        if isinstance(m1, nn.Conv2d):
            m1.weight.data = conv_first.weight.data.clone()
        elif isinstance(m1, nn.BatchNorm2d):
            m1.weight.data = BN_last.weight.data.clone()           
            m1.bias.data = BN_last.bias.data.clone()
            m1.running_mean = BN_last.running_mean.clone()
            m1.running_var = BN_last.running_var.clone()
        elif isinstance(m1, nn.Linear):
            m1.weight.data = linear_last.weight.data.clone()
            m1.bias.data = linear_last.bias.data.clone()

    # copy downsample
    for name0, m0 in model.named_modules():
        # print(name0)
        if 'downsample' in name0:
            downsample_list.append(m0)
    assert(len(downsample_list) == 3)

    for name1, m1 in newmodel.named_modules(): 
        # print(name1)
        if 'downsample' in name1:
            m0 = downsample_list.pop(0)
            m1.weight.data = m0.weight.data.clone()
            assert(m1.bias == None)

    block_id_mod = 0
    for block0, block1 in zip(model.modules(), newmodel.modules()):
        if isinstance(block0, Bottleneck):
            if block_id_mod == block_id:
                # print('block_id_mod%d' % block_id_mod)
                # print('block_id')
                conv_id_mod = 0
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m0, nn.Conv2d):
                        if name0 != 'downsample':
                            # print('Conv2d')
                            # print(conv_id_mod, conv_id + 1)
                            if conv_id_mod == conv_id:
                                new_out_channels = m1.weight.data.shape[0]
                                weight_copy = m0.weight.data.abs().clone()
                                weight_copy = weight_copy.cpu().numpy()
                                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                                arg_max = np.argsort(L1_norm)
                                max_l1_idx = arg_max[::-1][:new_out_channels]
                                assert max_l1_idx.size == new_out_channels, "size of arg_max_rev not correct"

                                m1.weight.data = m0.weight.data[max_l1_idx.tolist(), :, :, :].clone()
                            elif conv_id_mod == conv_id+1:
                                m1.weight.data = m0.weight.data[:, max_l1_idx.tolist(), :, :].clone()
                            else:
                                m1.weight.data = m0.weight.data.clone()
                            conv_id_mod += 1
                    elif isinstance(m0, nn.BatchNorm2d):
                        # print('BatchNorm2d')
                        # print(conv_id_mod, conv_id + 1)
                        if conv_id_mod == conv_id + 1:
                            m1.weight.data = m0.weight.data[max_l1_idx.tolist()].clone()
                            m1.bias.data = m0.bias.data[max_l1_idx.tolist()].clone()
                            m1.running_mean = m0.running_mean[max_l1_idx.tolist()].clone()
                            m1.running_var = m0.running_var[max_l1_idx.tolist()].clone()
                        else:
                            m1.weight.data = m0.weight.data.clone()
                            m1.bias.data = m0.bias.data.clone()
                            m1.running_mean = m0.running_mean.clone()
                            m1.running_var = m0.running_var.clone()
            else: 
                # print('no block_id')
                for (name0, m0), (name1, m1) in zip(block0.named_children(), block1.named_children()):
                    if isinstance(m1, nn.Conv2d):
                        if name0 != 'downsample':
                            m1.weight.data = m0.weight.data.clone()
                    elif isinstance(m1, nn.BatchNorm2d):
                        m1.weight.data = m0.weight.data.clone()           
                        m1.bias.data = m0.bias.data.clone()
                        m1.running_mean = m0.running_mean.clone()
                        m1.running_var = m0.running_var.clone()
                # if 'downsample' in name0:
                #     print('downsample')
            block_id_mod += 1

def prune_filter_l1(newmodel, conv_id, conv_end_id, model, freeze_conv, freeze_linear):
    preserve = []
    prune = []
    linear = []
    conv_id_mod = 0
    new_out_channels = 0
    max_l1_idx = None
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            if conv_id_mod == conv_id:
                new_out_channels = m1.weight.data.shape[0]
                weight_copy = m0.weight.data.abs().clone()
                weight_copy = weight_copy.cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                arg_max = np.argsort(L1_norm)
                max_l1_idx = arg_max[::-1][:new_out_channels]
                assert max_l1_idx.size == new_out_channels, "size of arg_max_rev not correct"

                m1.weight.data = m0.weight.data[max_l1_idx.tolist(), :, :, :].clone()
                prune.append(m1)
            elif conv_id_mod == conv_id+1:
                m1.weight.data = m0.weight.data[:, max_l1_idx.tolist(), :, :].clone()
                prune.append(m1)
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.weight.requires_grad = not freeze_conv
                preserve.append(m1)
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_id_mod == conv_id:
                m1.weight.data = m0.weight.data[max_l1_idx.tolist()].clone()
                m1.bias.data = m0.bias.data[max_l1_idx.tolist()].clone()
                m1.running_mean = m0.running_mean[max_l1_idx.tolist()].clone()
                m1.running_var = m0.running_var[max_l1_idx.tolist()].clone()

                prune.append(m1)
            elif conv_id_mod == conv_id+1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                prune.append(m1)
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                m1.weight.requires_grad = not freeze_conv
                m1.bias.requires_grad = not freeze_conv
                preserve.append(m1)
            conv_id_mod += 1
        elif isinstance(m0, nn.Linear):
            # For the last conv layer, train it and the next fully-connected layer from scratch
            if conv_id < conv_end_id:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.weight.requires_grad = not freeze_linear
                m1.bias.requires_grad = not freeze_linear
                linear.append(m1)
            else:
                m1.weight.data = m0.weight.data[:, max_l1_idx.tolist()].clone()
                m1.bias.data = m0.bias.data.clone()

                prune.append(m1)
    return preserve, prune, linear
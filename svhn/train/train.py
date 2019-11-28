import argparse
import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_fp16(model, optimizer, epoch, device, train_loader, valid, valid_len, log_interval):
    end = time.time()
    model.train()
    avg_loss = 0.
    train_acc = 0.
    train_len = len(train_loader.dataset)
    if valid:
        train_len = train_len - valid_len
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), train_len,
        #         100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = avg_loss / train_len
    train_acc = train_acc / train_len
    print('Train Epoch: %d Time: %.2f Avg loss: %.4f Train_acc: %.4f' % 
        (epoch, time.time()-end, avg_loss, train_acc))
    return avg_loss, train_acc

def train(model, optimizer, epoch, device, train_loader, valid, valid_len, log_interval):
    end = time.time()
    model.train()
    avg_loss = 0.
    train_acc = 0.
    train_len = len(train_loader.dataset)
    if valid:
        train_len = train_len - valid_len
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), train_len,
        #         100. * batch_idx / len(train_loader), loss.item()))
    avg_loss = avg_loss / train_len
    train_acc = train_acc / train_len
    print('Train Epoch: %d Time: %.2f Avg loss: %.4f Train_acc: %.4f' % 
        (epoch, time.time()-end, avg_loss, train_acc))
    return avg_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def valid(model, device, valid_loader, valid_len):
    model.eval()
    valid_loss = 0
    correct = 0
    # target_nums_list = [0] * 10
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            # for t in target:
            #     target_nums_list[t] += 1
        print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
            valid_loss / valid_len, correct, valid_len,
            100. * correct / valid_len))
        # print('target_nums_list', target_nums_list)
    return correct / valid_len

def train_eval(model, device, train_loader, valid_len, valid=True):
    model.eval()
    loss = 0
    correct = 0
    train_len = len(train_loader.dataset)
    if valid:
        train_len = train_len - valid_len
    # target_nums_list = [0] * 10
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            # for t in target:
            #     target_nums_list[t] += 1
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
            loss / train_len, correct, train_len,
            100. * correct / train_len))
        # print('target_nums_list', target_nums_list)
    return correct / train_len


def get_optim_set(preserve, prune, linear, preserve_lr=0.001, prune_lr=0.001, linear_lr=0.001):
    optim_set = []
    for mod in preserve:
        optim_set.append({
            'params': mod.parameters(), 
            'lr': preserve_lr,
            'weight_decay': 0.0001
        })

    for mod in prune:
        optim_set.append({
            'params': mod.parameters(), 
            'lr': prune_lr,
            'weight_decay': 0.0001
        })

    for mod in linear:
        optim_set.append({
            'params': mod.parameters(), 
            'lr': linear_lr,
            'weight_decay': 0.0001
        })
    return optim_set

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

from __future__ import print_function
#pytorch packages
import math

#For plots
import matplotlib.pyplot as plt

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

from  models.imagenet.efficientnet_model import EfficientNet
import models.imagenet as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *
from utils.dropback import *
from utils.choose_best import best_accuracy

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
efficientnet_models_names = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
model_names = default_model_names + customized_models_names + efficientnet_models_names


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
                    choices=DATA_BACKEND_CHOICES)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--id', '--init-decay', default=0.9, type=float,
                    metavar='W', help='initial weight decay (default: 0.9)',
                    dest='init_decay')
parser.add_argument('--tw', '--track-weights', default=10000, type=int,
                    metavar='W', help='number of weights to track (default: 100000)',
                    dest='track_weights')
parser.add_argument('--drop-back', default='True', type=str,
                    help='Set to have Dropback optimzer.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')




def main():
    global args
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    print("#Weights to track '{}'".format(args.track_weights))
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    model = None
    if  args.arch.startswith('mobilenetv2'):
        model =  models.__dict__[args.arch](width_mult=args.width_mult)
    elif args.arch.startswith('efficientnet'):
        print('efficientnet')
        model = EfficientNet.from_name(args.arch)
    else:
        model = models.__dict__[args.arch]()


    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    #print(model)
    #print(sum([param.nelement() for param in model.parameters()]))


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.drop_back == 'True':
        print("DropBack Optimizer is used!")
        optimizer = Dropback(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, track_size=args.track_weights, init_decay=args.init_decay)
    else:
        print("Baseline Optimizer is used!")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    cudnn.benchmark = True

    # Data loading code
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    
    logs,losses = find_lr(train_loader, train_loader_len, model, criterion, optimizer)
    plt.plot(logs[10:-5],losses[10:-5])
    plt.savefig("find_lr.png")
    print("done")
    #print(logs)
    #print(losses)

    print("Filtered--------------------------------")
    #print(logs[10:-5])
    #print(losses[10:-5])



def find_lr(train_loader, train_loader_len, model, criterion, optimizer, init_value= 1e-3, final_value= 1e+1, beta= 0.98):
    fast_factor=10
    num = (train_loader_len-1)/fast_factor
    print("To try:",num)
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for i,data in enumerate(train_loader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item() #.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #print("batch:",batch_num," ,lr:",math.log10(lr)," ,loss:",smoothed_loss)
        #print(batch_num,", ", end='')
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        if i > num:
            return log_lrs, losses
    return log_lrs, losses


if __name__ == '__main__':
    main()

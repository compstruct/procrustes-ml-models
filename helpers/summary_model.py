from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torchvision.models as models
from  models.imagenet.efficientnet_model import EfficientNet
import models.imagenet as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *
from tensorboardX import SummaryWriter
from dropback import *

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

parser = argparse.ArgumentParser(description='PyTorch Model Summary')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

def main():
    global args
    args = parser.parse_args()
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


    print(model)
    print(sum([param.nelement() for param in model.parameters()]))

if __name__ == '__main__':
    main()


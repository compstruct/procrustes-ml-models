from __future__ import print_function

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

from  models.imagenet.efficientnet_model import EfficientNet
import models.imagenet as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *
from tensorboardX import SummaryWriter
from utils.choose_best import best_accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau
from  utils.specified_scheduler import *

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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--early-term', default=False, type=bool,
                    help='Set to compare the accuracy of parallel trainings and terminate if worst than others.')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 255],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--use_qe', default='True', type=str, help='enable quantile estimation')
parser.add_argument('--q-init', default=1e-2, type=float,
                    metavar='QI', help='q init (default: 1e-2)')
parser.add_argument('--q-step', default=1e-6, type=float,
                    metavar='QS', help='q step (default: 1e-6)')
parser.add_argument('--q-sf', default=False, type=bool,
                    metavar='QF', help='q sf (default: False)')

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')

lr_schedul = None

best_prec1 = 0
cp_logger=None #Logger(os.path.join(args.checkpoint, 'cp_log.txt'), title=title)

def main():
    global args, best_prec1, cp_logger, lr_schedul
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        #np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

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

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    model_weights_tot = sum([param.nelement() for param in model.parameters()])
    print(model)
    print(model_weights_tot)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
   
    if args.use_qe == 'True':
        target_quantile = 1 - (args.track_weights / model_weights_tot) 
        print('target quantile: ', target_quantile) 
    else:
        target_quantile = None
        print('Use exact sort without qe') 
                    

    print("Baseline Optimizer is used!")
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
   
    if args.lr_decay == 'specified':
        lr_schedul = lr_scheduled('schedule.csv') 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False, threshold=1e-1, threshold_mode='abs', cooldown=0, min_lr=0.002, eps=1e-8)

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
            cp_logger = Logger(os.path.join(args.checkpoint, 'cp_log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            cp_logger = Logger(os.path.join(args.checkpoint, 'cp_log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
            cp_logger.set_names(['itteration in epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Train Acc. top5', 'Valid Acc.', 'Valid Acc. top5', 'batch speed'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        cp_logger = Logger(os.path.join(args.checkpoint, 'cp_log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        cp_logger.set_names(['itteration in epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Train Acc. top5', 'Valid Acc.', 'Valid Acc. top5', 'batch speed'])


    cudnn.benchmark = True

    # Data loading code
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)

    if args.evaluate:
        from collections import OrderedDict
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            source_state = torch.load(args.weight)
            target_state = OrderedDict()
            for k, v in source_state.items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            model.load_state_dict(target_state)
        else:
            print("=> no weight found at '{}'".format(args.weight))

        validate(val_loader, val_loader_len, model, criterion)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    print("Training started!", flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # the folder name to dump logs and files
        folder = 'baseline'

        best_reported_acc = best_accuracy(current_epoch=epoch, run_name='ImageNet_'+args.arch, folder=folder)
        print("Best reported Valid Acc. for epoch: "+str(epoch)+" is: "+str(best_reported_acc), flush=True)
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs), flush=True)

        # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        if (epoch > 5) & (epoch % 5 == 4): # save every 5 epoch!
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False, checkpoint=args.checkpoint, filename=str(epoch+1)+'_'+'checkpoint.pth.tar')

        if args.lr_decay == 'plateau2':
            scheduler.step(val_loss)

        # Early terminations
        print("Accuracy here: "+str(prec1) + " vs. "+str(best_reported_acc)+" best accuracy reported!", flush=True)
        
        if early_terminate(epoch, prec1, best_reported_acc):
            print("Early Termination on epoch "+str(epoch))
            break

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    cp_logger.close()
    #cp_logger.plot()
    #savefig(os.path.join(args.checkpoint, 'cp_log.eps'))

    writer.close()

    print('Best accuracy:')
    print(best_prec1, flush=True)

def early_terminate(epoch, prec1, best_reported_acc):
  
    if not args.early_term:
        #print("Will not check for early terminate")
        return False

    print("Will check for early terminate!")

    if prec1 > best_reported_acc:
        return False

#    if epoch < 1: # how many epochs to skip
#        if prec1 < (best_reported_acc/2):
#           return True
#        return False

    betha = 0.5
    alpha = 1 #4 # which epoch to start matching exactly

    coef = 1
    if epoch < alpha:
        coef = 1 - betha*( (alpha-epoch)/alpha ) # after alpha epochs, the 
    if prec1 < coef * best_reported_acc:
        return True
    else:
        return False

def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):
    global cp_logger
    bar = Bar('Processing', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if not args.lr_decay == 'plateau2':
            adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #cp_logger.append([i, losses.avg, 0, top1.avg, top1.avg, 0, 0, batch_time.avg])

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=train_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def validate(val_loader, val_loader_len, model, criterion):
    global cp_logger
    bar = Bar('Processing', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # itteration logger
        #cp_logger.append([i, 0, losses.avg, 0, 0, top1.avg, top1.avg, batch_time.avg])

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=val_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

last_epoch_lr_set = -1
from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter, monitor=None):
    global last_epoch_lr_set
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    elif args.lr_decay == 'fixedStep': # every n (scheduled) epoch, decay
        s = args.step # one element only
        count = round(epoch / s)
        lr = args.lr * pow(args.gamma, count)
    elif args.lr_decay == 'fixedStep2': # every n (scheduled) epoch, decay
        s = args.step
        if last_epoch_lr_set < epoch:
            last_epoch_lr_set = epoch
            if epoch == 0: # skip the first epoch and don't decay
                lr = args.lr
            elif (epoch % s) == 0: #5%3==2: 0,1,2,(3),4,5,(6) ...
                lr = lr * args.gamma
    elif args.lr_decay == 'plateau':
        if last_epoch_lr_set < epoch:
            last_epoch_lr_set = epoch
#            reduce_plat.on_epoch_end(epoch, monitor)
    elif args.lr_decay == 'specified':
        if last_epoch_lr_set < epoch:
            last_epoch_lr_set = epoch
            lr = lr_schedul.get_lr(epoch)
            print('read ' , lr, ' for epoch ', epoch)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

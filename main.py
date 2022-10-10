import os
import time
import argparse
import torch
from torch import nn
from torch.backends import cudnn
import dataset as dataset
import numpy as np
import torchvision
import torch.nn.functional as F
from wideresnet import WideResNet
from lenet import LeNet
import logging
import copy

parser = argparse.ArgumentParser(description='Revisiting Consistency Regularization for Deep Partial Label Learning')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--lam', default=1, type=float)

parser.add_argument('--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100', 'fmnist', 'kmnist'],
                    default='cifar10')

parser.add_argument('--model', type=str, choices=['widenet','lenet'], default='widenet')

parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--rate', default=0.4, type=float,help='-1 for feature, 0.x for random')

parser.add_argument('--trial', default='1', type=str)

parser.add_argument('--data-dir', default='./data/', type=str)

args = parser.parse_args()
best_prec1 = 0
num_classes = 10 if args.dataset != 'cifar100' else 100

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.StreamHandler()
                    ])

args.model_name = 'trial_{}_{}_dataset_{}_binomial_{}_lam_{}'.format(args.trial, args.model, args.dataset, args.rate, args.lam)
logging.info(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def DPLL_train(train_loader, model, optimizer, epoch, consistency_criterion, confidence):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    for i, (x_aug0, x_aug1, x_aug2, y, part_y, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # partial label
        part_y = part_y.float().cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        y_pred_aug0 = model(x_aug0)
        # augmentation1
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        # augmentation2
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)

        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

        # consist loss
        consist_loss0 = consistency_criterion(y_pred_aug0_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss1 = consistency_criterion(y_pred_aug1_probas_log, torch.tensor(confidence[index]).float().cuda())
        consist_loss2 = consistency_criterion(y_pred_aug2_probas_log, torch.tensor(confidence[index]).float().cuda())
        # supervised loss
        super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * (1 - part_y), dim=1))
        # dynamic lam
        lam = min((epoch / 100) * args.lam, args.lam)

        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + super_loss

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # update confidence
        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index)

        losses.update(final_loss.item(), x_aug0.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'lam ({lam})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lam=lam))

    return losses.avg


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(num_classes, 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()


def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(valid_loader), batch_time=batch_time, loss=losses,top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg


def DPLL():
    global args, best_prec1
    # load data
    if args.dataset == "cifar10":
        train_loader, test = dataset.cifar10_dataloaders(args.data_dir, args.rate)
        channel = 3
    elif args.dataset == 'svhn':
        train_loader, test = dataset.svhn_dataloaders(args.data_dir, args.rate)
        channel = 3
    elif args.dataset == 'cifar100':
        train_loader, test = dataset.cifar100_dataloaders(args.data_dir, args.rate)
        channel = 3
    elif args.dataset == 'fmnist':
        train_loader, test = dataset.fmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    elif args.dataset == 'kmnist':
        train_loader, test = dataset.kmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    else:
        assert "Unknown dataset"

    # load model
    if args.model == 'widenet':
        model = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    elif args.model == 'lenet':
        model = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
    else:
        assert "Unknown model"
    model = model.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    cudnn.benchmark = True
    # init confidence
    confidence = copy.deepcopy(train_loader.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    # Train loop
    for epoch in range(0, args.epochs):
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        # training
        trainloss = DPLL_train(train_loader, model, optimizer, epoch, consistency_criterion, confidence)
        # lr_step
        scheduler.step()
        # evaluate on validation set
        valacc, valloss = validate(test, model, criterion, epoch)


if __name__ == '__main__':
    DPLL()

# classification into (no cancer / cancer)

import argparse
import os
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchnet
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from munch import Munch
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import dataset


# pytorch 1.0 and torchvision 0.1.9
import torchvision
assert '0.1.9' in torchvision.__file__


def accuracy(output, target):
    pred = output.max(1)[1]
    return 100.0 * target.eq(pred).float().mean()


def save_checkpoint(checkpoint_dir, state, epoch):
    file_path = os.path.join(checkpoint_dir, 'checkpoint_{:08d}.pth.tar'.format(epoch))
    torch.save(state, file_path)
    return file_path


def adjust_learning_rate(optimizer, epoch):
    lr = cfg.optimizer.lr
    for e in cfg.optimizer.lr_decay_epochs:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    auc = torchnet.meter.AUCMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        acc = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        accuracies.update(acc, input.size(0))
        prob = nn.Softmax(dim=1)(output.detach())[:, 1].cpu().numpy()
        auc.add(prob, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, accuracy=accuracies))

    return batch_time.avg, data_time.avg, losses.avg, accuracies.avg, auc.value()[0]


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    auc = torchnet.meter.AUCMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            acc = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            accuracies.update(acc, input.size(0))
            prob = nn.Softmax(dim=1)(output)[:, 1].cpu().numpy()
            auc.add(prob, target)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.training.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, accuracy=accuracies))

    return batch_time.avg, losses.avg, accuracies.avg, auc.value()[0]


def main(cfg):
    if cfg.training.resume is not None:
        log_dir = cfg.training.log_dir
        checkpoint_dir = os.path.dirname(cfg.training.resume)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        log_dir = os.path.join(cfg.training.logs_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        checkpoint_dir = os.path.join(cfg.training.checkpoints_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    print("=> creating model '{}'".format(cfg.arch.model))
    model = models.__dict__[cfg.arch.model](pretrained=cfg.arch.pretrained)

    if cfg.arch.model.startswith('alexnet') or cfg.arch.model.startswith('vgg'):
        model.classifier._modules['6'] = nn.Linear(4096, cfg.arch.num_classes)
    elif cfg.arch.model == 'inception_v3':
        model.aux_logits = False
        model.fc = nn.Linear(2048, cfg.arch.num_classes)
    elif cfg.arch.model == 'resnet152':
        model.fc = nn.Linear(2048, cfg.arch.num_classes)
    else:
        raise Exception

    if cfg.arch.model.startswith('alexnet') or cfg.arch.model.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.optimizer.lr,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.weight_decay)

    start_epoch = 0
    if cfg.training.resume is not None:
        if os.path.isfile(cfg.training.resume):
            print("=> loading checkpoint '{}'".format(cfg.training.resume))
            checkpoint = torch.load(cfg.training.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.training.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.training.resume))
            print('')
            raise Exception

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = []
    val_transforms = []
    if cfg.arch.model == 'inception_v3':
        train_transforms.append(transforms.Scale(299))
        val_transforms.append(transforms.Scale(299))

    train_dataset = dataset.DDSM(cfg.data.root, 'train', transforms.Compose(train_transforms + [
        transforms.ToTensor(),
        normalize,
    ]))
    val_dataset = dataset.DDSM(cfg.data.root, 'val', transforms.Compose(val_transforms + [
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.workers, pin_memory=True)

    train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
    for epoch in range(start_epoch, cfg.training.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train_summary_writer.add_scalar('learning_rate', lr, epoch + 1)

        train_batch_time, train_data_time, train_loss, train_accuracy, train_auc = train(
            train_loader, model, criterion, optimizer, epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch + 1)
        train_summary_writer.add_scalar('auc', train_auc, epoch + 1)

        val_batch_time, val_loss, val_accuracy, val_auc = validate(
            val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)
        val_summary_writer.add_scalar('accuracy', val_accuracy, epoch + 1)
        val_summary_writer.add_scalar('auc', val_auc, epoch + 1)

        if (epoch + 1) % cfg.training.checkpoint_epochs == 0:
            checkpoint_path = save_checkpoint(checkpoint_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, epoch + 1)
            cfg.training.log_dir = log_dir
            cfg.training.resume = checkpoint_path
            with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
                f.write(cfg.toYAML())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', metavar='PATH', help='path to config file')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    main(cfg)

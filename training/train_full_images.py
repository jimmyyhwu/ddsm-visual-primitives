# classification into (no cancer / benign / cancer) on full images

import argparse
import os
import time
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchnet
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from munch import Munch
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import resnet


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root, image_list_path, target_size, transform):
        self.root = root
        name2class = {
            'normal': 0,
            'benign': 1,
            'cancer': 2,
        }
        with open(image_list_path, 'r') as f:
            self.images = [(line.strip(), name2class[line.strip()[:6]]) for line in f.readlines()]
        self.target_size = target_size
        self.transform = transform

        weight = np.unique([label for _, label in self.images], return_counts=True)[1]
        self.weight = 1 / (weight / np.amin(weight))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name, ground_truth = self.images[idx]
        image = Image.open(os.path.join(self.root, image_name))
        min_dim = min(image.size)
        ratio = self.target_size / min_dim
        new_size = (int(ratio * image.size[0]), int(ratio * image.size[1]))
        image = image.resize(new_size, resample=Image.BILINEAR)  # image shape is now (~1500, 896)
        delta_w = self.target_size - new_size[0]
        delta_h = self.target_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        image = ImageOps.expand(image, padding)
        image = np.asarray(image)
        image = np.broadcast_to(np.expand_dims(image, 2), image.shape + (3,))  # image shape is now (~1500, 896, 3)
        image = self.transform(image)  # image shape is now (3, ~1500, 896) and a it is a tensor
        return image, ground_truth


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
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
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
    auc0 = torchnet.meter.AUCMeter()
    auc1 = torchnet.meter.AUCMeter()
    auc2 = torchnet.meter.AUCMeter()
    model.train()
    end = time.time()
    for i, (input_data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = Variable(input_data)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), input_data.size(0))
        accuracies.update(acc, input_data.size(0))
        prob = nn.Softmax(dim=1)(output)
        auc0.add(prob.data[:, 0], target.eq(0))
        auc1.add(prob.data[:, 1], target.eq(1))
        auc2.add(prob.data[:, 2], target.eq(2))

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

    return batch_time.avg, data_time.avg, losses.avg, accuracies.avg, auc0.value()[0], auc1.value()[0], auc2.value()[0]


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    auc0 = torchnet.meter.AUCMeter()
    auc1 = torchnet.meter.AUCMeter()
    auc2 = torchnet.meter.AUCMeter()
    model.eval()
    end = time.time()

    for i, (input_data, target) in enumerate(val_loader):

        with torch.no_grad():
            target = target.cuda(async=True)
            input_var = Variable(input_data)
            target_var = Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), input_data.size(0))
        accuracies.update(acc, input_data.size(0))
        prob = nn.Softmax(dim=1)(output)
        auc0.add(prob.data[:, 0], target.eq(0))
        auc1.add(prob.data[:, 1], target.eq(1))
        auc2.add(prob.data[:, 2], target.eq(2))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, accuracy=accuracies))

    return batch_time.avg, losses.avg, accuracies.avg, auc0.value()[0], auc1.value()[0], auc2.value()[0]


def main():
    if cfg.training.resume is not None:
        log_dir = cfg.training.log_dir
        checkpoint_dir = os.path.dirname(cfg.training.resume)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        log_dir = os.path.join(cfg.training.logs_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        checkpoint_dir = os.path.join(cfg.training.checkpoints_dir,
                                      '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    print("=> creating model 'resnet152'")
    model = resnet.resnet152(pretrained=cfg.arch.pretrained)
    model.fc = nn.Linear(2048, cfg.arch.num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
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
    target_size = 512
    raw_image_dir = '../data/ddsm_raw'
    image_list_train = '../data/ddsm_raw_image_lists/train.txt'
    image_list_val = '../data/ddsm_raw_image_lists/val.txt'

    train_dataset = DDSM(raw_image_dir, image_list_train, target_size, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    val_dataset = DDSM(raw_image_dir, image_list_val, target_size, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(train_dataset.weight).float()).cuda()

    if cfg.training.debug:
        # limit training data for debugging:
        train_indices = range(cfg.data.batch_size * 10)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.workers, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
            num_workers=cfg.data.workers, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
    else:
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

        train_batch_time, train_data_time, train_loss, train_accuracy, train_auc0, train_auc1, train_auc2 = train(
            train_loader, model, criterion, optimizer, epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch + 1)
        train_summary_writer.add_scalar('auc0', train_auc0, epoch + 1)
        train_summary_writer.add_scalar('auc1', train_auc1, epoch + 1)
        train_summary_writer.add_scalar('auc2', train_auc2, epoch + 1)

        val_batch_time, val_loss, val_accuracy, val_auc0, val_auc1, val_auc2 = validate(
            val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)
        val_summary_writer.add_scalar('accuracy', val_accuracy, epoch + 1)
        val_summary_writer.add_scalar('auc0', val_auc0, epoch + 1)
        val_summary_writer.add_scalar('auc1', val_auc1, epoch + 1)
        val_summary_writer.add_scalar('auc2', val_auc2, epoch + 1)

        if (epoch + 1) % cfg.training.checkpoint_epochs == 0:
            checkpoint_path = save_checkpoint(checkpoint_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1)
            cfg.training.log_dir = log_dir
            cfg.training.resume = checkpoint_path
            with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
                f.write(cfg.toYAML())
            print("Checkpoint written: " + checkpoint_path)

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', metavar='PATH', help='path to config file')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as config_file:
        cfg = Munch.fromYAML(config_file)
    startTime = datetime.now()
    main()
    print("Runtime: %s" % (datetime.now() - startTime))

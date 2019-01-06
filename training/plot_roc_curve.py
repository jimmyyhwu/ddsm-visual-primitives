import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchnet
from munch import Munch
from torch.autograd import Variable

from dataset import DDSM
from models.resnet_3class import get_resnet152_3class_model

from tqdm import tqdm as tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --- Load config file: ---
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='path to config file')
args = parser.parse_args()
config_path = args.config_path
with open(config_path, 'r') as config_file:
    cfg = Munch.fromYAML(config_file)

# --- Load checkpoint: ---
checkpoint_path = None
if cfg.training.resume is not None:
    if os.path.isfile(cfg.training.resume):
        checkpoint_path = cfg.training.resume
    else:
        print("=> no checkpoint found at '{}'".format(cfg.training.resume))
        print('')
        raise Exception

# --- Prepare model: ---
model, start_epoch, optimizer_state, features_layer = get_resnet152_3class_model(checkpoint_path)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=cfg.optimizer.lr,
                            momentum=cfg.optimizer.momentum,
                            weight_decay=cfg.optimizer.weight_decay)

# --- Prepare dataset: ---
val_dataset = DDSM.create_full_image_dataset('val')
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
    num_workers=cfg.data.workers, pin_memory=True)

# --- Prepare data structures for results: ---
targets = [[] for _ in range(cfg.arch.num_classes)]
probs = [[] for _ in range(cfg.arch.num_classes)]
aucs = [torchnet.meter.AUCMeter() for _ in range(cfg.arch.num_classes)]

# --- Run model on all full images and store classification results: ---
for input, target in tqdm(val_loader):
    with torch.no_grad():
        input_var = Variable(input)
        output = model(input_var)
        prob = nn.Softmax(dim=1)(output)

    for i in range(cfg.arch.num_classes):
        aucs[i].add(prob[:, i].data, target == i)
        targets[i].extend(target.numpy() == i)
        probs[i].extend(prob[:, i].data.cpu().numpy())

# --- Print statistics: ---
for i in range(cfg.arch.num_classes):
    print('class {}'.format(i))
    print('torchnet.meter.AUCMeter: {}'.format(aucs[i].value()[0]))
    print('sklearn.metrics.roc_auc_score: {}'.format(roc_auc_score(targets[i], probs[i])))
    print('')


def plot_roc_curve(y_true, y_score, filename):
    plt.figure(figsize=(8, 8))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect('equal')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, color='black')
    plt.savefig(filename)


for i in range(cfg.arch.num_classes):
    checkpoint_identifier = checkpoint_path.replace('/', '__')
    plot_roc_curve(targets[i], probs[i], 'roc_for_class_{}_{}.png'.format(i, checkpoint_identifier))

# determine the units most frequently 'influential' to classification decisions

import argparse
import os
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from munch import Munch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
from PIL import Image

import datasets
import fcn_resnet


# pytorch 1.0 and torchvision 0.1.9
import torchvision
assert '0.1.9' in torchvision.__file__


def main(args):
    with open(args.config_path, 'r') as f:
        cfg = Munch.fromYAML(f)

    print("=> creating model '{}'".format(cfg.arch.model))
    if cfg.arch.model == 'resnet152':
        model = fcn_resnet.resnet152(num_classes=cfg.arch.num_classes)
        features_layer = model.layer4
    else:
        raise Exception

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    resume_path = cfg.training.resume.replace(cfg.training.resume[-16:-8], '{:08}'.format(args.epoch))
    resume_path = os.path.join('../training', resume_path)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(resume_path))

    # convert fc to conv
    model.module.surgery()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    patch_size = 227
    val_dataset = datasets.DDSM(args.raw_image_dir, args.raw_image_list_path, 'val', patch_size, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    # extract features and max activations
    features = []
    def feature_hook(module, input, output):
        features.extend(output.data.cpu().numpy())
    features_layer._forward_hooks.clear()
    features_layer.register_forward_hook(feature_hook)
    prob_maps = []
    max_class_probs = []
    with torch.no_grad():
        for _, image in tqdm(val_dataset):
            input = image.unsqueeze(0)
            input = input.cuda()
            output = model(input)
            output = output.transpose(1, 3).contiguous()
            size = output.size()[:3]
            output = output.view(-1, output.size(3))
            prob = nn.Softmax(dim=1)(output)
            prob = prob.view(size[0], size[1], size[2], -1)
            prob = prob.transpose(1, 3)
            prob = prob.cpu().numpy()
            prob_map = prob[0]
            prob_maps.append(prob_map)

    # save final fc layer weights
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy().squeeze(3).squeeze(2)

    # rank the units by influence
    max_activations = np.array([feature_map.max(axis=(1, 2)) for feature_map in features])
    max_activations = np.expand_dims(max_activations, 1)
    weighted_max_activations = max_activations * weight_softmax
    unit_indices = np.argsort(-weighted_max_activations, axis=2)
    all_unit_indices_and_counts = []
    for class_index in range(cfg.arch.num_classes):
        num_top_units = 8
        unit_indices_and_counts = list(zip(*np.unique(unit_indices[:, class_index, :num_top_units].ravel(), return_counts=True)))
        unit_indices_and_counts.sort(key=lambda x: -x[1])
        all_unit_indices_and_counts.append(unit_indices_and_counts)    

    # save rankings to file
    unit_rankings_dir = os.path.join(args.output_dir, 'unit_rankings', cfg.training.experiment_name, args.final_layer_name)
    if not os.path.exists(unit_rankings_dir):
        os.makedirs(unit_rankings_dir)
    with open(os.path.join(unit_rankings_dir, 'rankings.pkl'), 'wb') as f:
        pickle.dump(all_unit_indices_and_counts, f)

    # print some statistics
    for class_index in range(cfg.arch.num_classes):
        print('class index: {}'.format(class_index))
        # which units show up in the top num_top_units all the time?
        # note: unit_id == unit_index + 1
        num_top_units = 8
        unit_indices_and_counts = list(zip(*np.unique(unit_indices[:, class_index, :num_top_units].ravel(), return_counts=True)))
        unit_indices_and_counts.sort(key=lambda x: -x[1])

        # if we annotate the num_units_annotated top units, what percent of
        # the top num_top_units units on all val images will be annotated?
        num_units_annotated = 20
        print(unit_indices_and_counts[:num_units_annotated])
        annotated_count = sum(x[1] for x in unit_indices_and_counts[:num_units_annotated])
        unannotated_count = sum(x[1] for x in unit_indices_and_counts[num_units_annotated:])
        assert annotated_count + unannotated_count == num_top_units * len(features)
        print('percent annotated: {:.2f}%'.format(100.0 * annotated_count / (annotated_count + unannotated_count)))
        print('')


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='../training/pretrained/resnet152_3class/config.yml')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--final_layer_name', default='layer4')
parser.add_argument('--raw_image_dir', default='../data/ddsm_raw')
parser.add_argument('--raw_image_list_path', default='../data/ddsm_raw_image_lists/val.txt')
parser.add_argument('--output_dir', default='output/')
args = parser.parse_args()
main(args)

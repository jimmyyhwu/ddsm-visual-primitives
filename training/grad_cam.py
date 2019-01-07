# parts from https://github.com/kazuto1011/grad-cam-pytorch
from __future__ import print_function
from collections import OrderedDict
import os
import cv2
import numpy as np
import torch#
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F


def run_grad_cam(image_path, cuda):
    CONFIG = {
        'resnet152': {
            'target_layer': 'module.layer4.2',
            'input_size': 224
        },
    }

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # Synset words
    classes = list()
    with open('../training/samples/diagnosis.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)


    model = get_resnet152_3class_model()

    # Image
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (CONFIG['resnet152']['input_size'],) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image).unsqueeze(0)

    print('Grad-CAM')

    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image.to(device))

    for i in range(0, 3):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['resnet152']['target_layer'])
        save_gradcam('static/results/{}_gcam_{}.png'.format(classes[idx[i]], 'resnet152'), output, raw_image)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


def get_resnet152_3class_model(checkpoint_path='../training/checkpoint_resnet152_deepminer_epoch_50.pth.tar'):
    print("=> creating model 'resnet152'")
    model = models.__dict__['resnet152'](pretrained=not checkpoint_path)
    model.fc = torch.nn.Linear(2048, 3)
   # features_layer = model.layer4

    model = torch.nn.DataParallel(model)
   # cudnn.benchmark = True

    epoch = 50
    optimizer_state = None
    if checkpoint_path:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path,  map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        optimizer_state = checkpoint['optimizer']
        epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    else:
        print("=> no checkpoint loaded, only ImageNet weights")

    return model


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.detach().cpu().numpy()

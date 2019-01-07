# do a forward pass on all full images in validation set and store results in DB
import argparse

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from munch import Munch
from torch.autograd import Variable
from common.dataset import preprocess_image_default
from common.model import get_model_from_config

import sys
sys.path.insert(0, '..')


def run_image_through_model(model, features_layer, image_path):
    image = preprocess_image_default(image_path)
    image_batch = image.unsqueeze(0)  # unsqueeze: (3, ~1500, 896) -> (1, 3, ~1500, 896)

    print("run image through model")
    # extract features and max activations
    feature_maps = []
    feature_maps_maximums = []

    def feature_hook(_, __, layer_output):  # args: module, input, output
        # layer_output.data.shape: (2048, ~50, 28)
        nonlocal feature_maps, feature_maps_maximums
        feature_maps = layer_output.data.cpu().numpy()[0]
        feature_maps_maximums = feature_maps.max(axis=(1, 2))  # shape: (2048)  //only the max value of one activation matrix. This allows to sort them in a list

    features_layer._forward_hooks.clear()
    features_layer.register_forward_hook(feature_hook)

    with torch.no_grad():
        input_var = Variable(image_batch)
        output = model(input_var)  # forward pass with hooks
        class_probs = nn.Softmax(dim=1)(output).squeeze(0)  # shape: [3], i.e. [0.9457, 0.0301, 0.0242]
        classification = int(np.argmax(class_probs.cpu().numpy()))  # int

    print("saved max activations per unit")
    return feature_maps_maximums, feature_maps, classification


def create_unit_ranking_for_one_image(model, feature_maps_maximums, feature_maps):
    print("create unit ranking")

    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()

    # rank the units by influence
    feature_maps_maximums = np.expand_dims(feature_maps_maximums, 0)  # shape: (1, 2048)
    weighted_max_activations = feature_maps_maximums * weight_softmax  # shape: (num_classes=3, 2048)

    units_and_activations = []

    for unit_id, influence_per_class in enumerate(weighted_max_activations.T):         # 2048, number of units
        units_and_activations.append((unit_id, influence_per_class, feature_maps[unit_id]))

    return units_and_activations


def return_top_units(units_and_activations, diagnosis, number_of_units):
    print("Sort for top", number_of_units, "units")
    ranked_units_and_activations = sorted(units_and_activations, key=lambda x: x[1][diagnosis], reverse=True)[:number_of_units]

    for idx, val in enumerate(ranked_units_and_activations):
        print(idx, val[0])
    # entries of ranked_units_and_activations: unit_name, diagnosis[0,1,2], activation_map for the unit
    return ranked_units_and_activations


def analyze_one_image(image_path, cfg=None):

    if not cfg:
        config_path = "/home/mp1819/ddsm-visual-primitives-python3/training/logs_full_images/2018-12-20_13-23-05.683938_resnet152/config.yml"
        with open(config_path, 'r') as f:
            cfg = Munch.fromYAML(f)

    model, features_layer, checkpoint_path = get_model_from_config(cfg)
    feature_maps_maximums, feature_maps, classification = run_image_through_model(model, features_layer, image_path)
    units_and_activations = create_unit_ranking_for_one_image(model, feature_maps_maximums, feature_maps)
    top_units_and_activations = return_top_units(units_and_activations, classification, 10)
    return top_units_and_activations


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    parser.add_argument('--image_path')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = Munch.fromYAML(f)

    analyze_one_image(args.image_path, cfg)


if __name__ == '__main__':
    test()


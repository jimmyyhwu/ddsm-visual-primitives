# do a forward pass on all full images in validation set and store results in DB

import argparse
import os
import pickle
import hashlib

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from munch import Munch
from torch.autograd import Variable
from tqdm import tqdm as tqdm

from dataset import DDSM
from models.resnet_3class import get_resnet152_3class_model
from db.database import DB


def prepare_model(cfg):
    resume_path = cfg.training.resume.replace(cfg.training.resume[-16:-8], '{:08}'.format(args.epoch))
    resume_path = os.path.join('../training', resume_path)
    model, epoch, optimizer_state, features_layer = get_resnet152_3class_model(resume_path)
    return model, features_layer, resume_path


def run_model_on_all_images(model, features_layer, dataset):
    # extract features and max activations
    max_activation_per_unit_per_input = []

    def feature_hook(_, __, layer_output):  # args: module, input, output
        # layer_output.date.shape: (2048, ~50, 28)
        feature_maps = layer_output.data.cpu().numpy()[0]
        feature_maps_maximums = feature_maps.max(axis=(1, 2))  # shape: (2048)
        max_activation_per_unit_per_input.append(feature_maps_maximums)

    features_layer._forward_hooks.clear()
    features_layer.register_forward_hook(feature_hook)

    classifications = []
    correct = 0

    i = 0
    for image, ground_truth in tqdm(dataset):
        with torch.no_grad():
            input_var = Variable(image.unsqueeze(0))  # unsqueeze: (3, ~1500, 896) -> (1, 3, ~1500, 896)
            output = model(input_var)  # shape: [1, 3]
            class_probs = nn.Softmax(dim=1)(output).squeeze(0)  # shape: [3], i.e. [0.9457, 0.0301, 0.0242]
            classification = int(np.argmax(class_probs.cpu().numpy()))  # int
            if classification == ground_truth:
                correct += 1
            i += 1

    print("\n", "Correct classified: %d Image count: %d, Ratio: %f" % (correct, i, correct / i))

    return max_activation_per_unit_per_input, classifications


def create_unit_ranking(model, max_activation_per_unit_per_input):
    # save final conv layer weights
    params = list(model.parameters())
    # params[-2].data.cpu().numpy().shape: (3, 2048)
    weight_softmax = params[-2].data.cpu().numpy()  # shape: (num_classes=3, 2048)

    # rank the units by influence
    max_activations = np.expand_dims(max_activation_per_unit_per_input, 1)  # shape: (input_count, 1, 2048)
    weighted_max_activations = max_activations * weight_softmax  # shape: (input_count, num_classes=3, 2048)
    # with np.argsort we essentially replace activations with unit_id in sorted order
    # (unit_id equals original activation index here)
    ranked_units = np.argsort(-weighted_max_activations, axis=2)
    unit_id_and_count_per_class = []
    for class_index in range(cfg.arch.num_classes):
        num_top_units = 8
        # we need a list like this: (top1_img_1 ... top8_img_1, top1_img_2 ... top8_img_2, ...)
        top_units_for_each_input = ranked_units[:, class_index, :num_top_units].ravel()
        # make each element a tuple like this: (unit_id, count of appearance in top8)
        unit_indices_and_counts = zip(*np.unique(top_units_for_each_input, return_counts=True))
        unit_indices_and_counts = sorted(unit_indices_and_counts, key=lambda x: -x[1])
        unit_id_and_count_per_class.append(unit_indices_and_counts)

    return unit_id_and_count_per_class, ranked_units, weighted_max_activations


def save_rankings_to_file(unit_id_and_count_per_class, args, cfg):
    unit_rankings_dir = os.path.join(args.output_dir, 'unit_rankings', cfg.training.experiment_name,
                                     args.final_layer_name)
    if not os.path.exists(unit_rankings_dir):
        os.makedirs(unit_rankings_dir)
    with open(os.path.join(unit_rankings_dir, 'rankings.pkl'), 'wb') as f:
        pickle.dump(unit_id_and_count_per_class, f)


def save_activations_to_db(weighted_max_activations, val_dataset, db_filename, checkpoint_path):
    db = DB(db_filename, "../db/")
    conn = db.get_connection()
    num_classes = 3
    image_names = val_dataset.image_names

    with open(checkpoint_path, 'rb') as f:
        network_hash = hashlib.md5(f.read()).hexdigest()

    insert_statement_net = "INSERT OR REPLACE INTO net (id, net, filename) VALUES (?, ?, ?)"
    conn.execute(insert_statement_net, (network_hash, 'resnet152', checkpoint_path))

    for class_index in range(num_classes):
        for image_index in range(len(image_names)):
            image_name = image_names[image_index]
            max_activation_per_unit = weighted_max_activations[image_index, class_index]
            temp = max_activation_per_unit.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(max_activation_per_unit))

            select_stmt_img = "SELECT id FROM image WHERE image_path = ?"
            c = conn.cursor()
            c.execute(select_stmt_img, (image_name,))
            row = c.fetchone()
            if row is None:
                print("Error: Image is not in database: %s" % image_name)
                break
            img_id = row[0]

            for unit_index in range(len(max_activation_per_unit)):
                activation = max_activation_per_unit[unit_index]
                rank = ranks[unit_index]

                insert_statement = "INSERT OR REPLACE INTO image_unit_activation (net_id, image_id, unit_id, class_id, activation, rank) VALUES (?, ?, ?, ?, ?, ?)"
                conn.execute(insert_statement, (network_hash, img_id, unit_index + 1, class_index, float(activation), int(rank)))

    conn.commit()


def print_statistics(ranked_units, max_activation_per_unit_per_input):

    print("\nSome statistics:\n")

    for class_index in range(cfg.arch.num_classes):
        print('class index: {}'.format(class_index))
        # which units show up in the top num_top_units all the time?
        # note: unit_id == unit_index + 1
        num_top_units = 8
        unit_indices_and_counts = zip(*np.unique(ranked_units[:, class_index, :num_top_units].ravel(),
                                                 return_counts=True))
        unit_indices_and_counts = sorted(unit_indices_and_counts, key=lambda x: -x[1])

        # if we annotate the num_units_annotated top units, what percent of
        # the top num_top_units units on all val images will be annotated?
        num_units_annotated = 20
        print(unit_indices_and_counts[:num_units_annotated])
        annotated_count = sum(x[1] for x in unit_indices_and_counts[:num_units_annotated])
        unannotated_count = sum(x[1] for x in unit_indices_and_counts[num_units_annotated:])
        assert annotated_count + unannotated_count == num_top_units * len(max_activation_per_unit_per_input)
        print('percent annotated: {:.2f}%'.format(100.0 * annotated_count / (annotated_count + unannotated_count)))
        print('')


def analyze_full_images(args, cfg, db_path):
    model, features_layer, checkpoint_path = prepare_model(cfg)
    val_dataset = DDSM.create_full_image_dataset('val')

    max_activation_per_unit_per_input, classifications = run_model_on_all_images(model, features_layer, val_dataset)
    unit_id_and_count_per_class, ranked_units, weighted_max_activations = create_unit_ranking(model, max_activation_per_unit_per_input)

    save_rankings_to_file(unit_id_and_count_per_class, args, cfg)
    save_activations_to_db(weighted_max_activations, val_dataset, db_path, checkpoint_path)

    print_statistics(ranked_units, max_activation_per_unit_per_input)


if __name__ == "__main__":
    DB_FILENAME = os.environ['DB_FILENAME'] if 'DB_FILENAME' in os.environ else 'test.db'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='../training/pretrained/resnet152_3class/config.yml')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--final_layer_name', default='layer4')
    parser.add_argument('--raw_image_dir', default='../data/ddsm_raw')
    parser.add_argument('--raw_image_list_path', default='../data/ddsm_raw_image_lists/val.txt')
    parser.add_argument('--output_dir', default='output/')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = Munch.fromYAML(f)

    analyze_full_images(args, cfg, DB_FILENAME)

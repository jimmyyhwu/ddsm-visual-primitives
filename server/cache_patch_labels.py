import argparse
import os
import pickle


def process_line(line):
    image_name, label = line.strip().split(' ')
    label = int(label)
    return image_name, label


parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', default='../data/ddsm_2class/val.txt')
parser.add_argument('--output_dir', default='data/labels/')
parser.add_argument('--output_file', default='val.pickle')
args = parser.parse_args()

with open(args.labels_path, 'r') as f:
    image_list = map(process_line, f.readlines())

cache = {}
for image_path, label in image_list:
    _, image_name = os.path.split(image_path)
    cache[image_name] = label

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
    pickle.dump(cache, f)

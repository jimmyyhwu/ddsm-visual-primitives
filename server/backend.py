import base64
import glob
import os
import pickle

from PIL import Image

STATIC_DIR = 'static'
DATA_DIR = 'data'
LOG_DIR = os.path.join(DATA_DIR, 'log')


def get_models_and_layers(full=False, ranked=False):
    if full:
        unit_vis_dir = os.path.join(STATIC_DIR, 'unit_vis')
    else:
        unit_vis_dir = os.path.join(STATIC_DIR, 'unit_vis_subset')
    models_and_layers = []
    models = sorted(os.listdir(unit_vis_dir))
    for model in models:
        layers = sorted(os.listdir(os.path.join(unit_vis_dir, model)))
        for layer in layers:
            rankings_path = 'data/unit_rankings/{}/{}/rankings.pkl'.format(model, layer)
            if not ranked or os.path.exists(rankings_path):
                models_and_layers.append((model, layer))
    return models_and_layers


def get_responded_units(name):
    responses = get_responses(name)
    all_units = []
    for (model, layer) in get_models_and_layers(full=True):
        layer_dir = os.path.join('unit_vis', model, layer)
        model_and_layer = '{}/{}'.format(model, layer)
        units = []
        for unit in sorted(os.listdir(os.path.join(STATIC_DIR, layer_dir))):
            key = '{}/{}/{}'.format(model, layer, unit)
            if key in responses.keys():
                units.append(unit)
        if len(units) > 0:
            all_units.append((model_and_layer, units))
    return all_units


def get_units(name, model, layer, sample=8, full=False, ranked=False):
    if full:
        layer_dir = os.path.join('unit_vis', model, layer)
    else:
        layer_dir = os.path.join('unit_vis_subset', model, layer)
    responses = get_responses(name)
    units = []
    sums = []
    labels = get_labels()
    label_symbols = {0: '-', 1: 'o', 2: '+'}
    for unit in sorted(os.listdir(os.path.join(STATIC_DIR, layer_dir))):
        key = '{}/{}/{}'.format(model, layer, unit)
        message = '[response recorded]' if key in responses else ''
        unit_dir = os.path.join(layer_dir, unit)
        image_names = sorted(os.listdir(os.path.join(STATIC_DIR, unit_dir)))[:sample]
        unit_labels = [labels[image_name[5:]] for image_name in image_names]
        unit_labels_str = ' '.join([label_symbols[label] for label in unit_labels])
        image_paths = [os.path.join(unit_dir, x) for x in image_names]
        units.append((unit, message, image_paths, unit_labels_str))
        sums.append(sample - sum(unit_labels))
    if ranked:
        num_units_per_class = 20
        ranked_units = []
        rankings_path = 'data/unit_rankings/{}/{}/rankings.pkl'.format(model, layer)
        with open(rankings_path, 'rb') as f:
            rankings = pickle.load(f)
        for class_index, unit_rankings in enumerate(rankings):
            for unit_index, count in unit_rankings[:num_units_per_class]:
                unit = units[unit_index]
                ranked_units.append((unit[0], unit[1], unit[2],
                                     '(class {}, count {}) {}'.format(class_index, count, unit[3])))
        units = ranked_units
    else:
        sums, units = zip(*sorted(zip(sums, units)))
    return list(units)


def get_unit_data(name, model, layer, unit, sample=32, num_cols=4):
    unit_dir = os.path.join('unit_vis', model, layer, unit)
    image_names = sorted(os.listdir(os.path.join(STATIC_DIR, unit_dir)))[:sample]
    entries = []
    labels = get_labels()
    label_names = {0: 'normal', 1: 'begnin', 2: 'malignant'}
    for image_name in image_names:
        # remove the first 5 chars containing the image rank
        parts = image_name[5:].split('-')
        label = label_names[labels[image_name[5:]]]
        raw_name = '{}-{}.jpg'.format(parts[0], parts[1])
        raw_width, raw_height = Image.open(os.path.join(STATIC_DIR, 'raw/{}'.format(raw_name))).size
        # noinspection PyTypeChecker
        box = parts[2].split('_')
        x = int(box[1][1:])
        y = int(box[0][1:])
        w = int(box[3][1:])
        h = int(box[2][1:])
        assert w == h
        height = 100 * float(h) / raw_height
        width = 100 * float(w) / raw_width
        left = 100 * float(x) / raw_width
        top = 100 * float(y) / raw_height
        style = 'position: absolute; height: {}%; width: {}%; left: {}%; top: {}%;'.format(height, width, left, top)
        entry = {
            'img_name': 'unit_vis/{}/{}/{}/{}'.format(model, layer, unit, image_name),
            'style': style,
            'raw_name': 'raw/{}'.format(raw_name),
            'label': label
        }
        entries.append(entry)
    data = [entries[i:i + num_cols] for i in range(0, len(entries), num_cols)]
    responses = get_responses(name)
    old_response = None
    key = '{}/{}/{}'.format(model, layer, unit)
    if key in responses:
        old_response = responses[key]
    return data, old_response


def get_labels():
    labels = {}
    for labels_path in glob.glob('data/labels/*.pickle'):
        print(labels_path)
        with open(labels_path, 'rb') as f:
            labels.update(pickle.load(f))
    return labels


def log_response(data):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    timestamp = data['timestamp']
    filename = '{}_response.pickle'.format(timestamp)
    with open(os.path.join(LOG_DIR, filename), 'wb') as f:
        pickle.dump(data, f)


def get_responses(name):
    encoded_name = base64.urlsafe_b64encode(bytes(name, 'utf-8'))
    data_path = os.path.join(DATA_DIR, '{}.pickle'.format(encoded_name))
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            responses = pickle.load(f)
    else:
        responses = {}
    return responses


def get_num_responses(name):
    return len(get_responses(name).keys())


def store_response(name, model, layer, unit, data):
    responses = get_responses(name)
    key = '{}/{}/{}'.format(model, layer, unit)
    responses[key] = data
    encoded_name = base64.urlsafe_b64encode(bytes(name, 'utf-8'))
    data_path = os.path.join(DATA_DIR, '{}.pickle'.format(encoded_name))
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    with open(data_path, 'wb') as f:
        pickle.dump(responses, f)


def get_summary():
    summary = []
    for pickle_path in sorted(glob.glob(os.path.join(DATA_DIR, '*.pickle'))):
        with open(pickle_path, 'rb') as f:
            responses = pickle.load(f)
        name = responses.itervalues().next()['name']
        responded_units = get_responded_units(name)
        summary.append((name, responded_units))
    return summary

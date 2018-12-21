from datetime import datetime

from flask import Flask, render_template, request
from flask import redirect
import urllib
import urllib.parse
import os
import backend
import subprocess
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '../deepminer')
import analyze_full_images

app = Flask(__name__)

STATIC_DIR = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'processed')
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')
app.config['ACTIVATIONS_FOLDER'] = PROCESSED_FOLDER

@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/summary')
def summary():
    response_summary = backend.get_summary()
    summary2 = []
    for (name, responded_units) in response_summary:
        unquote_name = urllib.parse.unquote_plus(name)
        summary2.append((name, unquote_name, responded_units))
    return render_template('summary.html', summary=summary2)


@app.route('/handle_login', methods=['POST'])
def handle_login():
    name = request.form['name']
    name = urllib.parse.quote_plus(name)
    return redirect('/home/{}'.format(name))


@app.route('/home/<name>')
def home(name):
    unquote_name = urllib.parse.unquote_plus(name)
    models_and_layers = backend.get_models_and_layers()
    models_and_layers = map(lambda x: '{}/{}'.format(x[0], x[1]), models_and_layers)
    full_models_and_layers = backend.get_models_and_layers(full=True)
    full_models_and_layers = map(lambda x: '{}/{}'.format(x[0], x[1]), full_models_and_layers)
    ranked_models_and_layers = backend.get_models_and_layers(full=True, ranked=True)
    ranked_models_and_layers = map(lambda x: '{}/{}'.format(x[0], x[1]), ranked_models_and_layers)
    responded_units = backend.get_responded_units(name)
    num_responses = backend.get_num_responses(name)
    return render_template('home.html', name=name, unquote_name=unquote_name, num_responses=num_responses,
                           models_and_layers=models_and_layers, full_models_and_layers=full_models_and_layers,
                           ranked_models_and_layers=ranked_models_and_layers,
                           responded_units=responded_units)


@app.route('/overview/<name>/<model>/<layer>')
def overview(name, model, layer, full=False, ranked=False):
    unquote_name = urllib.parse.unquote_plus(name)
    units = backend.get_units(name, model, layer, full=full, ranked=ranked)
    num_responses = backend.get_num_responses(name)
    if ranked:
        full = False
    return render_template('overview.html', name=name, unquote_name=unquote_name, num_responses=num_responses,
                           full=full, ranked=ranked, model=model, layer=layer, units=units)


@app.route('/overview/full/<name>/<model>/<layer>')
def overview_full(name, model, layer):
    return overview(name, model, layer, full=True)


@app.route('/overview/ranked/<name>/<model>/<layer>')
def overview_ranked(name, model, layer):
    return overview(name, model, layer, full=True, ranked=True)


@app.route('/survey/<name>/<model>/<layer>/<unit>')
def survey(name, model, layer, unit, full=False, ranked=False):
    unquote_name = urllib.parse.unquote_plus(name)
    data, old_response = backend.get_unit_data(name, model, layer, unit)
    num_responses = backend.get_num_responses(name)
    return render_template('survey.html', name=name, unquote_name=unquote_name, num_responses=num_responses, full=full,
                           ranked=ranked, model=model, layer=layer, unit=unit, data=data, old_response=old_response)


@app.route('/survey/full/<name>/<model>/<layer>/<unit>')
def survey_full(name, model, layer, unit):
    return survey(name, model, layer, unit, full=True)


@app.route('/survey/ranked/<name>/<model>/<layer>/<unit>')
def survey_ranked(name, model, layer, unit):
    return survey(name, model, layer, unit, ranked=True)


@app.route('/handle_survey', methods=['POST'])
def handle_survey(full=False, ranked=False):
    name = request.form['name']
    model = request.form['model']
    layer = request.form['layer']
    unit = request.form['unit']
    data = {q: response for q, response in request.form.iteritems()}
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    data['timestamp'] = timestamp
    backend.log_response(data)
    backend.store_response(name, model, layer, unit, data)
    if ranked:
        return redirect('/overview/ranked/{}/{}/{}#{}'.format(name, model, layer, unit))
    elif full:
        return redirect('/overview/full/{}/{}/{}#{}'.format(name, model, layer, unit))
    else:
        return redirect('/overview/{}/{}/{}#{}'.format(name, model, layer, unit))


@app.route('/handle_survey/full', methods=['POST'])
def handle_survey_full():
    return handle_survey(full=True)


@app.route('/handle_survey/ranked', methods=['POST'])
def handle_survey_ranked():
    return handle_survey(ranked=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(full_path)

    return render_template('single_image.html', success=True, full_path=full_path)


@app.route('/process_image', methods=['POST'])
def process_image():
    original_path = os.path.join(app.config['UPLOAD_FOLDER'],  'test.jpg')
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'benign.jpg')
    top_units_and_activations = analyze_full_images.analyze_one_image()

    activation_map_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation.jpg')

    activation_map = top_units_and_activations[0][2] # activation map for unit 0 => top unit
    print(activation_map)

    # resize activation map to img size
    img = Image.open(original_path)
    basewidth = img.size[0]
    wpercent = (basewidth / float(len(activation_map[0])))
    hsize = int((float(len(activation_map[1])) * float(wpercent)))

    activation_map_resized = np.resize(activation_map, (basewidth, hsize))
    plt.imsave(activation_map_path, activation_map_resized)

    activations_overlayed_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'benign.jpg')
    img.save(activations_overlayed_path)


    return render_template('single_image.html', success=False, processed=True, full_path=processed_path,
                           top_units_and_activations=top_units_and_activations,
                           activation_map_path=activation_map_path, activations_overlayed_path=activations_overlayed_path)


@app.route('/single_image/')
def single_image():
    return render_template('single_image.html', success=False, processed=False, full_path='')


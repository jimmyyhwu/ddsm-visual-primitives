from datetime import datetime

from flask import Flask, render_template, request
from flask import redirect
import urllib
import urllib.parse
import os
import backend
import sys
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '../training')
from analyze_single_image import SingleImageAnalysis
from common.dataset import get_preview_of_preprocessed_image

single_image_analysis = SingleImageAnalysis()

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
   # backend.register_doctor_if_not_exists(name)
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
    response_data = dict(request.form)  # create a mutable dictionary copy
    response_data['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    backend.log_response(response_data)

    name = request.form['name']  # doctor username
    model = request.form['model']  # resnet152
    layer = request.form['layer']  # layer4
    unit = request.form['unit']   #unit_0076
    q1 = request.form['q1']
    q2 = request.form['q2']
    q3 = request.form['q3']
    q4 = request.form['q4']
    answers = (q1, q2, q3, q4)
    backend.store_response(name, model, layer, unit, answers, response_data)
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
    result = single_image_analysis.analyze_one_image(os.path.join('../server/static/uploads', 'benign.jpg'))

    activation_map_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation.jpg')

    top_units_and_activations = result.get_top_units(result.classification, 1)
    activation_map = top_units_and_activations[0][2]  # activation map for unit 0 => top unit

    img = Image.open(original_path)
    # normalize activation values between 0 and 255
    activation_map_normalized = backend.normalize_activation_map(activation_map)

    # resize activation map to img size
    activation_map_resized = backend.resize_activation_map(img, activation_map_normalized)

    plt.gray()  # grayscale
    plt.imsave(activation_map_path, activation_map_resized)

    activations_overlayed_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'benign.jpg')
    img.save(activations_overlayed_path)

    backend.grad_cam()
    return render_template('single_image.html', success=False, processed=True, full_path=processed_path,
                           top_units_and_activations=top_units_and_activations,
                           activation_map_path=activation_map_path, activations_overlayed_path=activations_overlayed_path)


@app.route('/single_image/')
def single_image():
    return render_template('single_image.html', success=False, processed=False, full_path='')


@app.route('/example_analysis')
def example_analysis():
    preprocessed_full_image_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'full_image.jpg')

    image_path = '../data/ddsm_raw/cancer_05-C_0128_1.LEFT_CC.LJPEG.1.jpg'
    preprocessed_full_image = get_preview_of_preprocessed_image(image_path)
    preprocessed_image_height = preprocessed_full_image.size[1]
    preprocessed_full_image.save(preprocessed_full_image_path)
    result = single_image_analysis.analyze_one_image(image_path)

    units_to_show = 10
    top_units_and_activations = result.get_top_units(result.classification, units_to_show)

    for i in range(units_to_show):
        activation_map = top_units_and_activations[i][2]  # activation map for unit with rank i

        activation_map_normalized = backend.normalize_activation_map(activation_map)

        act_map_img = Image.fromarray(activation_map_normalized.astype(np.uint8), mode="L")
        act_map_img = ImageOps.colorize(act_map_img, (0, 0, 0), (255, 0, 0))
        act_map_img = act_map_img.resize(preprocessed_full_image.size, resample=Image.BICUBIC)
        activation_map_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation_{}.jpg'.format(i))
        act_map_img.save(activation_map_path, "JPEG")

    activation_map_prefix = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation_')
    return render_template('example_analysis.html',
                           image_path=result.image_path,
                           preprocessed_full_image_path=preprocessed_full_image_path,
                           preprocessed_image_height=preprocessed_image_height,
                           checkpoint_path=result.checkpoint_path,
                           classification=result.classification,
                           class_probs=result.class_probs,
                           top_units_and_activations=top_units_and_activations,
                           activation_map_prefix=activation_map_prefix)


@app.route('/unit/<unit_id>')
def unit(unit_id):

    top_images = backend.get_top_images_for_unit(unit_id)

    return render_template('unit.html',
                           unit_id=unit_id,
                           top_images=top_images)


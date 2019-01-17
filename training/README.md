
# CNN Training and Evaluation

## Dependencies

- python 3.6
- PIL
- pytorch 1.0
- torchvision 0.1.9 (`conda install torchvision=0.1.9 -c soumith`)
- [torchnet](https://github.com/pytorch/tnt) (`pip install git+https://github.com/pytorch/tnt.git@master`)
- tensorboardX
- tensorflow (for running TensorBoard)
- pyyaml
- munch
- jupyter
- matplotlib
- scipy
- scikit-learn
- tqdm

You should be able to use `pip` to install most of these dependencies.

Note that our models were trained on 227x227 input images using torchvision 0.1.9. This [commit](https://github.com/pytorch/vision/commit/df7524f2623126af25c6edd43d6e82110d502b69), incorporated into torchvision 0.2.0, requires ResNet inputs to be 224x224. To reproduce our reported numbers, please use torchvision 0.1.9 and 227x227 input images.

## Setup

Please reference the main README for instructions to download our processed DDSM data. For data processing details, please see our paper.

To reproduce the numbers reported in the paper, you will need to use the [`download_pretrained.sh`](download_pretrained.sh) script to download our pretrained models:

```bash
./download_pretrained.sh
```
The pretrained model checkpoints will be downloaded to the `pretrained` directory.

## Usage

### Training

The [`train.py`](train.py) and [`train_3class.py`](train_3class.py) training scripts use Munch config files for starting and resuming training runs. An initial config file is used to start a run, and an updated config file is written to the run's log directory every time a checkpoint is written. The updated config file contains all the parameters required to resume the run from the last checkpoint.

Please see the [`config`](config) directory for the reference configurations we used. For example, the following command trains an AlexNet for (no cancer / cancer) classification:

```bash
python train.py config/2class/alexnet.yml
```

A ResNet-152 for (no cancer / benign/ cancer) classification can also be trained:

```bash
python train_3class.py config/3class/resnet152.yml
```

The `logs_dir` parameter specifies the root log directory, which defaults to `logs`. Each run will create a new log directory inside `logs_dir` for storing both TensorBoard log files for visualization and Munch config files for resuming the run. Similarly, the `checkpoints_dir` parameter specifies the root checkpoint directory, which defaults to `checkpoints`. Each run creates a new checkpoint directory for storing model checkpoints. It may be appropriate to point `checkpoints_dir` to a location with more storage capacity.

To resume a training run from the latest checkpoint, simply pass in the config file from the log directory for that run. For example:

```bash
python train_2class.py logs/run_dir/config.yml
```

You may also modify the `resume` parameter to resume from an earlier checkpoint. Note that no config file will be written to the run's log directory until the first model checkpoint is written.

### Monitoring

The training script periodically logs loss, accuracy, and AUC for visualization in TensorBoard. To start TensorBoard, point it at the root log directory:

```bash
tensorboard --logdir logs/
```

Then navigate to `localhost:6006` in your web browser.

### Evaluation

Please see the [`plot_roc_curve.ipynb`](plot_roc_curve.ipynb) notebook for code to compute AUCs and plot ROC curves for trained models. First start a Jupyter notebook in the current directory:

```bash
jupyter notebook
```

Then open the [`plot_roc_curve.ipynb`](plot_roc_curve.ipynb) notebook. To evaluate the model from a specific run, you will need to modify `config_path` to point at the config file in the run's log directory. You will also need to specify `epoch` to indicate which checkpoint epoch to evaluate.

We provide the models used in our paper along with the corresponding config files, which you can download using the [`download_pretrained.sh`](download_pretrained.sh) script. You can uncomment the corresponding lines in the notebook to evaluate those models.

# CNN Training and Evaluation

## Setup

Please reference the main [README](../README.md) for instructions to download our processed DDSM data. For data processing details, please see our papers.

To reproduce the numbers reported in our papers, you will need to use the [`download-pretrained.sh`](download-pretrained.sh) script to download our pretrained models:

```bash
./download-pretrained.sh
```
The pretrained model checkpoints will be downloaded to the `pretrained` directory.

## Usage

### Training

We trained 2-class (no cancer / cancer) models for our SPIE paper, and 3-class models (no cancer / benign / cancer) for our HDSR paper.

The [`train_2class.py`](train_2class.py) and [`train_3class.py`](train_3class.py) training scripts use Munch config files for starting and resuming training runs. An initial config file is used to start a run, and an updated config file is written to the run's log directory every time a checkpoint is written. The updated config file contains all the parameters required to resume the run from the last checkpoint.

Please see the [`config`](config) directory for the reference configurations we used. For example, the following command trains an AlexNet for (no cancer / cancer) classification:

```bash
python train.py config/alexnet_2class.yml
```

The following command trains a 3-class ResNet-152 for (no cancer / benign / cancer) classification:

```bash
python train_3class.py config/resnet152_3class.yml
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

We provide the models used in our paper along with the corresponding config files, which you can download using the [`download-pretrained.sh`](download-pretrained.sh) script. You can uncomment the corresponding lines in the notebook to evaluate those models.

### Additional Notes

* Our models were all trained on 227x227 input images using [torchvision](https://pytorch.org/vision/stable/index.html) 0.1.9. This [commit](https://github.com/pytorch/vision/commit/df7524f2623126af25c6edd43d6e82110d502b69), incorporated into torchvision 0.2.0, requires ResNet inputs to be 224x224, so newer versions of torchvision are not compatible with our pretrained models. To reproduce our reported numbers, please use torchvision 0.1.9 and 227x227 input images.
* Due to a [bug](https://discuss.pytorch.org/t/inception3-runtimeerror-the-expanded-size-of-the-tensor-3-must-match-the-existing-size-864-at-non-singleton-dimension-3/32090) in torchvision's `inception_v3` that was still present as of version 0.1.9, we include our own copy of `inception_v3` in [`inception.py`](inception.py) with the bug fixed.

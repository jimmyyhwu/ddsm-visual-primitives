# CNN Unit Visualization

This code adapts the [reference code](https://github.com/metalbubble/cnnvisualizer/blob/master/pytorch_generate_unitsegments.py) for the paper *Object Detectors Emerge in Deep Scene CNNs* ([arXiv](https://arxiv.org/abs/1412.6856)) to visualize the internal units of trained CNNs.

## Usage

The [`generate_unitsegments.py`](generate_unitsegments.py) script takes in a trained model's config file and a checkpoint epoch to use. The script uses the validation split of our DDSM data. To generate unit visualizations for the same models we used, please go to [`training`](../training/), download the pretrained models, and then use the following commands:

```bash
# alexnet_2class
python generate_unitsegments.py --experiment_name alexnet_2class --config_path ../training/pretrained/alexnet_2class/config.yml --epoch 45

# vgg16_2class
python generate_unitsegments.py --experiment_name vgg16_2class --config_path ../training/pretrained/vgg16_2class/config.yml --epoch 12

# inception_v3_2class
python generate_unitsegments.py --experiment_name inception_v3_2class --config_path ../training/pretrained/inception_v3_2class/config.yml --epoch 7

# resnet152_2class
python generate_unitsegments.py --experiment_name resnet152_2class --config_path ../training/pretrained/resnet152_2class/config.yml --epoch 5

# resnet152_3class
python generate_unitsegments.py --experiment_name resnet152_3class --config_path ../training/pretrained/resnet152_3class/config.yml --epoch 5
```

The output visualizations will be written to `output/images/`, in a subdirectory specified by `--experiment_name`. Visualizations are created for all units in each convolutional layer. By default, the top 64 images for each unit are visualized. For example, `output/images/alexnet/conv5/unit_0001` would contain 64 images corresponding to visualizations of the top 64 images for unit_0001 in the conv5 layer of a trained AlexNet.

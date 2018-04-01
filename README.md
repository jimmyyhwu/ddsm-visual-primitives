
# ddsm-visual-primitives

This code release accompanies the following paper:

### Expert identification of visual primitives used by CNNs during mammogram classification [[arXiv](https://arxiv.org/abs/1803.04858)]

Jimmy Wu, Diondra Peck, Scott Hsieh, Vandana Dialani, Constance D. Lehman, Bolei Zhou, Vasilis Syrgkanis, Lester Mackey, Genevieve Patterson

*SPIE Medical Imaging 2018*

**Abstract:** This work interprets the internal representations of deep neural networks trained for classification of diseased tissue in 2D mammograms. We propose an expert-in-the-loop interpretation method to label the behavior of internal units in convolutional neural networks (CNNs). Expert radiologists identify that the visual patterns detected by the units are correlated with meaningful medical phenomena such as mass tissue and calcificated vessels. We demonstrate that several trained CNN models are able to produce explanatory descriptions to support the final classification decisions. We view this as an important first step toward interpreting the internal representations of medical classification CNNs and explaining their predictions.

## Overview

Directory | Purpose
------|--------
[`data`](data) | DDSM data
[`training`](training) | CNN training and evaluation code
[`unit_visualization`](unit_visualization) | CNN unit visualization code
[`server`](server) | Flask server code for expert annotation web interface

## Getting Started

You can download our processed DDSM data (about 15GB total) using the following script:

```bash
./download_data.sh
```

Please see the [`training`](training) directory for CNN training and evaluation code. We provide pretrained models to reproduce the numbers reported in the paper.

To run the annotation web interface, you will need to use a trained CNN to generate unit visualizations using code in [`unit_visualization`](unit_visualization), then start a web server using code in [`server`](server).

Please reference additional READMEs in the respective directories for more detailed instructions.

## Citation

If you find our work useful for your research, please consider citing:

```
@proceeding{doi: 10.1117/12.2293890,
author = {Jimmy Wu, Diondra Peck, Scott Hsieh, Vandana Dialani, Constance D. Lehman, Bolei Zhou, Vasilis Syrgkanis, Lester Mackey, Genevieve Patterson},
title = {Expert identification of visual primitives used by CNNs during mammogram classification},
journal = {Proc.SPIE},
volume = {10575},
pages = {10575 - 10575 - 9},
year = {2018},
doi = {10.1117/12.2293890},
URL = {https://doi.org/10.1117/12.2293890},
}
```


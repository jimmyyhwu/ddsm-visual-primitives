
# ddsm-visual-primitives

This code release accompanies the following two papers:

### Expert identification of visual primitives used by CNNs during mammogram classification [[arXiv](https://arxiv.org/abs/1803.04858)]

Jimmy Wu, Diondra Peck, Scott Hsieh, Vandana Dialani, Constance D. Lehman, Bolei Zhou, Vasilis Syrgkanis, Lester Mackey, Genevieve Patterson

*SPIE Medical Imaging 2018*

**Abstract:** This work interprets the internal representations of deep neural networks trained for classification of diseased tissue in 2D mammograms. We propose an expert-in-the-loop interpretation method to label the behavior of internal units in convolutional neural networks (CNNs). Expert radiologists identify that the visual patterns detected by the units are correlated with meaningful medical phenomena such as mass tissue and calcificated vessels. We demonstrate that several trained CNN models are able to produce explanatory descriptions to support the final classification decisions. We view this as an important first step toward interpreting the internal representations of medical classification CNNs and explaining their predictions.

### DeepMiner: Discovering Interpretable Representations for Mammogram Classification and Explanation [[arXiv](https://arxiv.org/abs/1805.12323)]

Jimmy Wu, Bolei Zhou, Diondra Peck, Scott Hsieh, Vandana Dialani, Lester Mackey, Genevieve Patterson

**Abstract:** We propose DeepMiner, a framework to discover interpretable representations in deep neural networks and to build explanations for medical predictions. By probing convolutional neural networks (CNNs) trained to classify cancer in mammograms, we show that many individual units in the final convolutional layer of a CNN respond strongly to diseased tissue concepts specified by the BI-RADS lexicon. After expert annotation of the interpretable units, our proposed method is able to generate explanations for CNN mammogram classification that are correlated with ground truth radiology reports on the DDSM dataset. We show that DeepMiner not only enables better understanding of the nuances of CNN classification decisions, but also possibly discovers new visual knowledge relevant to medical diagnosis.

<img src="figure.png" width="100%">

## Overview

Directory | Purpose
------|--------
[`data`](data) | DDSM data
[`deepminer`](deepminer) | Code for DeepMiner
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
@inproceedings{wu2018expert,
  title={Expert identification of visual primitives used by CNNs during mammogram classification },
  author={Wu, Jimmy and Peck, Diondra and Hsieh, Scott and Dialani, Vandana and Lehman, Constance D. and Zhou, Bolei and Syrgkanis, Vasilis and Mackey, Lester and Patterson, Genevieve},
  booktitle={Proc. SPIE 10575, Medical Imaging 2018: Computer-Aided Diagnosis},
  year={2018},
  doi = {10.1117/12.2293890}
}
```

```
@article{wu2018deepminer,
  title={DeepMiner: Discovering Interpretable Representations for Mammogram Classification and Explanation},
  author={Wu, Jimmy and Zhou, Bolei and Peck, Diondra and Hsieh, Scott and Dialani, Vandana and Mackey, Lester and Patterson, Genevieve},
  journal={arXiv preprint arXiv:1805.12323},
  year={2018}
}
```

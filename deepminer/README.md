# DeepMiner

This code implements our proposed DeepMiner framework and applies it to the DDSM dataset.

## Usage

### Model Training

Please see the [`training`](../training) directory for instructions to train a 3-class ResNet-152. Alternatively, you can use our pretrained model (`resnet152_3class`). The instructions for downloading the pretrained model are [here](../training#setup).

### Unit Visualization

Please see the [`unit_visualization`](../unit_visualization) directory for instructions to generate unit visualizations for the trained 3-class ResNet-152.

### Unit Ranking

Please use [`generate_unit_rankings.py`](generate_unit_rankings.py) to rank the units. Since the final layer has 2048 units, it would be infeasible to annotate all of them, so we will find the most frequently influential units and annotate only those. If you are using our pretrained model, you can simply run:

```bash
python generate_unit_rankings.py
```

The script will write out unit rankings to the `output` directory. By default, we select the top 20 units for each of the 3 classes, for a total of 60 units to annotate.

### Unit Annotation

See the [`server`](../server) directory to start up the expert annotation web interface. The top 60 units as ranked using [`generate_unit_rankings.py`](generate_unit_rankings.py) will be displayed under the "alternative ranking" section on the home page. You can view the 46 unit annotations done by our experts in the notebook [`cleaned_unit_labels.ipynb`](cleaned_unit_labels.ipynb).

### DeepMiner Explanations

See the [`deepminer_explanations.ipynb`](deepminer_explanations.ipynb) notebook for code used to generate the DeepMiner explanations shown in our paper.

### Explanations Benchmark

We carried out an experiment to test whether the DeepMiner explanations enable an expert to better distinguish malignant cases from benign cases. All relevant code for that experiment is contained in [`explanations_benchmark.ipynb`](explanations_benchmark.ipynb).

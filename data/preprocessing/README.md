# Preprocess DDSM dataset to train DeepMiner classifier
The DeepMiner classifier operates on sub-patches of the DDSM images in order to capture local tissue phenomena indicating cancer. In order to use the DDSM dataset to train the network, use the included scripts to preprocess the dataset. First configure your local environment variables in a `config.sh` file. Use the `config.sh_example` as an example. 
```
# To create image patches and annotation masks.
save_ddsm_patch_datasets.py

# To split into training and test sets
# Set the flag to PREPROCESS_IMAGES=true to have the images be resized to 227x227
preprocess_ddsm_patches.sh 
```

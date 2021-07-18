#!/bin/bash
# Splits a DDSM patch data set into a training, validation,
# and test sets and stores training and validation sets in 
# an LMDB format that can directly be used by the deep learning 
# framework Caffe.
#
# Args:
#   LABEL_FILE: Input DDSM label file to split
#
# Tasks:
# - (optionally) resizes all images in IMAGES_DIRECTORY to 227x227 
#   pixels and saves them to PREPROCESS_IMAGES_DIRECTORY, preserving
#   the image format of the original images
# - divides LABEL_FILE into training, validation, and test sets
# - generates training and validation LMDBs corresponding to the datasets
# - generate mean image for background subtraction
#
# Example usage:
# ./preprocess_ddsm_patches.sh data/ddsm/patches/merged_labels/binary-any-malignant-imgfrac0.5_stridefrac0.5-breastpct0.5_patchpct0.3_lesionpct0.3.tsv

# Call the config file which initialize the local file structure variables
. config.sh

# Input DDSM label file to split
LABEL_FILE=$1
# Should images be preprocessed? (true or false)
PREPROCESS_IMAGES=false
# Proportion of exams that will be in the training and validation splits;
# all remaining exams will be in the test split
TRAIN_FRAC=0.8
VAL_FRAC=0.1
# Random number generator seed for train/val splitting
SEED=123
# Format of input images; will be preserved when images are resized
IMG_TYPE="jpg"
# Directory in which raw patch images are stored (images not organized into subdirectories)
IMAGES_DIRECTORY=data/ddsm/patches/images
# Directory where we will store the resized images that lmdb will take as input
# These will have the same format as the input images.
# This folder will have no subfolder substructure.
# Needed when calling the lmdb function. 
PREPROCESS_IMAGES_DIRECTORY=data/ddsm/patches/resized_images
# Director where to store the output of the lmdb command
LMDB_DIRECTORY=lmdb #data/ddsm/patches/lmdb

# We will now resize each input jpg image from the ddsm director into a 227x227 image
# and store it in the preprocess image directory. The convert and parallel function
# need to be installed in your terminal for this part of the code.  
if [ $PREPROCESS_IMAGES == true ]; then
    # Create preprocess directory if it does not exist
    mkdir -p $PREPROCESS_IMAGES_DIRECTORY

    echo "Resizing images and saving as $IMG_TYPE"
    python preprocess_ddsm_patch_images.py $IMAGES_DIRECTORY \
	$PREPROCESS_IMAGES_DIRECTORY jpg $IMG_TYPE 227 227
    echo "Images have been successfully saved to $PREPROCESS_IMAGES_DIRECTORY/."
fi

# Divide target patch dataset into training, val, and test sets
echo "Dividing $LABEL_FILE into training ($TRAIN_FRAC), val ($VAL_FRAC), and test sets"
python generate_ddsm_patch_train_val_test_sets.py \
        $LABEL_FILE \
	$TRAIN_FRAC \
	$VAL_FRAC \
	$SEED

# Create lmdb directory if it does not exist already
mkdir -p $LMDB_DIRECTORY

# The generate sets script output contains the names of the original images
# which are in jpg format. So here we just replace the ending of the filenames from jpg
# to png, since this is the name in the preprocess directory.

# Extract basename and path from LABEL_FILE
LABEL_FILE_BASE="$(basename $LABEL_FILE)"
LABEL_PATH="$(dirname $LABEL_FILE)"
# Get basename without extension
LABEL_FILE_BASE_SANS_EXT="${LABEL_FILE_BASE%.*}"

# Construct training and validation file names without extensions
TRAIN_LABEL_NAME="train-seed${SEED}_train${TRAIN_FRAC}_val${VAL_FRAC}-${LABEL_FILE_BASE_SANS_EXT}"
VAL_LABEL_NAME="val-seed${SEED}_train${TRAIN_FRAC}_val${VAL_FRAC}-${LABEL_FILE_BASE_SANS_EXT}"

# Create LMDBs for the training and validation sets
rm -R -f $LMDB_DIRECTORY/$TRAIN_LABEL_NAME
rm -R -f $LMDB_DIRECTORY/$VAL_LABEL_NAME
echo "Generating LMDB $TRAIN_LABEL_NAME"
$CAFFE_ROOT/build/tools/convert_imageset --backend=lmdb \
    --shuffle \
    $PREPROCESS_IMAGES_DIRECTORY/ \
    $LABEL_PATH/$TRAIN_LABEL_NAME.txt \
    $LMDB_DIRECTORY/$TRAIN_LABEL_NAME
echo "Generating LMDB $VAL_LABEL_NAME"
$CAFFE_ROOT/build/tools/convert_imageset --backend=lmdb \
    --shuffle \
    $PREPROCESS_IMAGES_DIRECTORY/ \
    $LABEL_PATH/$VAL_LABEL_NAME.txt \
    $LMDB_DIRECTORY/$VAL_LABEL_NAME

echo "Generating mean image for backgroud substraction"
$CAFFE_ROOT/build/tools/compute_image_mean $LMDB_DIRECTORY/$TRAIN_LABEL_NAME \
    $LMDB_DIRECTORY/mean_$TRAIN_LABEL_NAME.binaryproto

echo "Done"

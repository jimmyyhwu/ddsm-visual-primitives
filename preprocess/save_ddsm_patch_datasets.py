#!/usr/bin/env python
# This script generates DDSM image patches and associated label files
import os, sys
import mammogram
import dataloader
from joblib import Parallel, delayed
import multiprocessing

# Should parallelism be used in executing this script?
use_parallelism = True

# Folder with DDSM jpg files organized into subfolders by type 
# (e.g., cancers, normals, benigns, ...)
ddsm_jpg_dir = os.path.join('/data','ddsm','raw')
# Folder with DDSM overlay files organized into subfolders by type
# (e.g., cancers, normals, benigns, ...)
ddsm_overlay_dir = os.path.join('/data','ddsm','overlays')
# Folder to which patch datasets will be saved
ddsm_patch_dir = os.path.join('/data','ddsm','patches','raw')

# Should patch images be saved?
save_images = True
# Which extension should be used for saved images
extension = 'jpg'

# Which labels should be generated? (See mammogram.get_patch_label())
label_names = ['breast_pct',
    'patch_pct-any','patch_pct-mass','patch_pct-calc',
    'lesion_pct-any','lesion_pct-mass','lesion_pct-calc',
    'patch_pct-any-malignant','patch_pct-mass-malignant',
    'patch_pct-calc-malignant',
    'lesion_pct-any-malignant','lesion_pct-mass-malignant',
    'lesion_pct-calc-malignant',
    'patch_pct-any-3','patch_pct-mass-3','patch_pct-calc-3',
    'lesion_pct-any-3','lesion_pct-mass-3','lesion_pct-calc-3',
    'patch_pct-any-4','patch_pct-mass-4','patch_pct-calc-4',
    'lesion_pct-any-4','lesion_pct-mass-4','lesion_pct-calc-4',
    'patch_pct-any-5','patch_pct-mass-5','patch_pct-calc-5',
    'lesion_pct-any-5','lesion_pct-mass-5','lesion_pct-calc-5']

# Specify patch dataset settings as a series of parallel arrays.
# The i-th entry of each array corresponds to the i-th dataset generated.
# Tile width and height as fraction of image width and height 
img_fracs = [0.5, 0.25, 0.125]
# What fraction of minimum patch dimension used as stride
stride_fracs = [0.5, 0.5, 0.5]

# Load DDSM exams from specified folders
types = ['cancers','benigns','benign_without_callbacks','normals']
exams = dataloader.ddsm_load_data(ddsm_jpg_dir,ddsm_overlay_dir,types=types)

# Generate patch images and labels associated with each scan
# in each exam
def save_exam_patches(exam):
    for scan in exam.scans:
        scan.save_patch_datasets(ddsm_patch_dir, img_fracs, stride_fracs,
                                 label_names, save_images,
                                 extension)
num_cores = multiprocessing.cpu_count() if use_parallelism else 1
Parallel(n_jobs = num_cores)(delayed(save_exam_patches)(exam) for exam in exams)

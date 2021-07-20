import numpy as np
import pandas as pd
import sys
import os
import random
import glob
import fnmatch
import dicom
import scipy.misc
from joblib import Parallel, delayed
import multiprocessing

# It resizes the img to a size given by the tuple resize. It preserves 
# the aspect ratio of the initial img and pads the remaining pixels with 
# black background. 
def resize_image(img, resize, interp_method='bicubic'):
    # Check which dimension of the input img needs larger scaling down and 
    # define the size of the ouput image accordingly
    if(float(img.shape[0])/resize[0]>float(img.shape[1])/resize[1]):
        new_h = resize[0]
        new_w = int(np.floor(float(resize[0])*img.shape[1]/img.shape[0]))
    else:
        new_w = resize[1]
        new_h = int(np.floor(float(resize[1])*img.shape[0]/img.shape[1]))
    
    # Create a numpy array with size equal to resize and full of black background
    new_img = np.zeros(resize)
    # Set the upper corner of the image equal to the resized image
    if len(img.shape) == 2:
        new_img[0:new_h,0:new_w] = scipy.misc.imresize(img,(new_h,new_w),interp_method)
    else:
        new_img[0:new_h,0:new_w] = scipy.misc.imresize(img,(new_h,new_w),interp_method)[0:new_h,0:new_w,0]
    
    return new_img
            
def read_in_one_image(im_name, resize, interp_method='bicubic', normalize=True):
    try:
        type = im_name.split('.')[-1].lower()                
        # Check if it is a dicom image                
        if(type=='dcm'):
            dicom_content = dicom.read_file(im_name)
            img = dicom_content.pixel_array
        # Otherwise if it is jpg just read
        else:
            img = scipy.misc.imread(im_name)

        img = resize_image(img, resize, interp_method)

        # Normalize image
        img = img.astype(np.float32)
        if normalize:            
            img -= np.mean(img)
            img /= np.std(img)
        
        
        # check that img is in shape (n,m,3)
        if len(img.shape) == 2:
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
            img[0,0,0] = 0
            img[0,0,1] = 1
            img[0,0,2] = 2
        
        return img
    except IOError, e:
        print 'Could not open image file for {}'.format(self)
        return []

# Transforms the input name into an output name for where to save the processed
# image
def get_output_name(im_name, output_dir, output_img_type):
    basename = os.path.basename(im_name) 
    pre, ext = os.path.splitext(basename)
    png_name = pre+'.'+output_img_type
    return os.path.join(output_dir,png_name)
        
if __name__ == '__main__':
    # Folder containing input patch images (images should not be organized into 
    # subdirectories)
    input_dir = sys.argv[1]
    # Folder for target png images
    output_dir = sys.argv[2]
    # Type of input images
    input_img_type = sys.argv[3]
    # Type of output images
    output_img_type = sys.argv[4]
    # Size to which each image should be resized
    height = int(sys.argv[5])
    width = int(sys.argv[6])

    # Interpolation ?
    interp = sys.argv[7] or 'bilinear'
    # Normalize Image (z-score) ?
    norm = sys.argv[8] or False
    
    def read_and_save(im_name):
        out_name = get_output_name(im_name,output_dir,output_img_type)
        scipy.misc.imsave(out_name, read_in_one_image(im_name,(height,width)))
        print "Saved {}".format(out_name)
    
    num_cores = multiprocessing.cpu_count()
    inputs = Parallel(n_jobs = num_cores)(
        delayed(read_and_save)(im) for im in 
        glob.iglob(os.path.join(input_dir,"*.{}".format(input_img_type))))

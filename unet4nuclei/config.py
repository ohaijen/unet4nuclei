
## 01
## PREPROCESSING

config_vars = {}

# assume a nucleus is at least 10 by 10 pixels big
config_vars["min_nucleus_size"] = 25

# Transform gray scale TIF images to PNG
config_vars["transform_images_to_PNG"] = True

# Pixels of the boundary (min 2 pixels)
config_vars["boundary_size"] = 2

# INPUT DIRECTORIES
config_vars["dir_root"] = '/data1/image-segmentation/BBBC022/'

# raw data
config_vars["dir_raw_images"] = config_vars["dir_root"] + 'raw_images/'
config_vars["dir_raw_annotations"] = config_vars["dir_root"] + 'new_renamed_annotations/'

# Split files
config_vars["create_split_files"] = False
config_vars["path_files_training"] = config_vars["dir_root"] + 'training.txt'
config_vars["path_files_validation"] = config_vars["dir_root"] + 'validation.txt'
config_vars["path_files_test"] = config_vars["dir_root"] + 'test.txt'

# Maximum number of training images (0 for all)
config_vars["max_training_images"] = 25

# Output directories

## split folders
config_vars["dir_training"] = config_vars["dir_root"] + 'unet/split_25/training/'
config_vars["dir_validation"] = config_vars["dir_root"] + 'unet/split_25/validation/'
config_vars["dir_test"] = config_vars["dir_root"] + 'unet/split_25/test/'

## boundary output
config_vars["dir_boundary_labels"] = config_vars["dir_root"] + 'unet/x_split_25/'

## input data, normalized and 8 bit
config_vars["dir_images_normalized_8bit"] = config_vars["dir_root"] + 'unet/y_split_25/'

# Data Augmentation options (using elastic deformation)

# augmentation taks lots of times but only has to be computed once 
config_vars["augment_images"] =  False

# augmentation parameters 
config_vars["n_points"] = 16
config_vars["distort"] = 5

# number of augmented images
config_vars["n_augmentations"] = 10



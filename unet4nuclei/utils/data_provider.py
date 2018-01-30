import numpy as np
import os
import os.path
import keras.preprocessing.image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage.io

import utils.data_augmentation


def data_from_array(data_dir):
    
    # load  x
    training_x = np.load(data_dir+"training/x.npy")
    test_x = np.load(data_dir+"test/x.npy")
    validation_x = np.load(data_dir+"validation/x.npy")

    print(training_x.shape)
    print(test_x.shape)
    print(validation_x.shape)

    # normalize
    training_x = training_x / 255
    test_x = test_x / 255
    validation_x = validation_x / 255

    # load y
    training_y = np.load(data_dir+"training/y.npy")
    test_y = np.load(data_dir+"test/y.npy")
    validation_y = np.load(data_dir+"validation/y.npy")

    print(training_y.shape)
    print(test_y.shape)
    print(validation_y.shape)
    
    return [training_x, training_y, validation_x, validation_y, test_x, test_y]


def data_from_images(data_dir, batch_size, bit_depth, dim1, dim2):
    
    flow_train = single_data_from_images(data_dir + 'training/x/', data_dir + 'training/y/', batch_size, bit_depth, dim1, dim2)
    flow_validation = single_data_from_images(data_dir + 'validation/x', data_dir + 'validation/y', batch_size, bit_depth, dim1, dim2)
    flow_test = single_data_from_images(data_dir + 'test/x', data_dir + 'test/y', batch_size, bit_depth, dim1, dim2)
    
    return [flow_train, flow_validation, flow_test]


def single_data_from_images(x_dir, y_dir, batch_size, bit_depth, dim1, dim2, rescale_labels):

    rescale_factor = 1./(2**bit_depth - 1)
    
    if(rescale_labels):
        rescale_factor_labels = rescale_factor
    else:
        rescale_factor_labels = 1

    gen_x = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor)
    gen_y = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor_labels)
    
    seed = 42

    stream_x = gen_x.flow_from_directory(
        x_dir,
        target_size=(dim1,dim2),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )
    stream_y = gen_y.flow_from_directory(
        y_dir,
        target_size=(dim1,dim2),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )
    
    flow = zip(stream_x, stream_y)
    
    return flow


def single_data_from_images_1d_y(x_dir, y_dir, batch_size, bit_depth, dim1, dim2, rescale_labels):

    rescale_factor = 1./(2**bit_depth - 1)
    
    if(rescale_labels):
        rescale_factor_labels = rescale_factor
    else:
        rescale_factor_labels = 1

    gen_x = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor)
    gen_y = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor_labels)
    
    seed = 42

    stream_x = gen_x.flow_from_directory(
        x_dir,
        target_size=(dim1,dim2),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )
    stream_y = gen_y.flow_from_directory(
        y_dir,
        target_size=(dim1,dim2),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )
    
    flow = zip(stream_x, stream_y)
    
    return flow


def random_sample_generator(x_big_dir, y_big_dir, batch_size, bit_depth, dim1, dim2, rescale_labels):

    do_augmentation = True
    
    # get image names
    image_names = os.listdir(x_big_dir)
    print('Found',len(image_names), 'images.')

    # get dimensions right -- understand data set
    n_images = len(image_names)
    ref_img = skimage.io.imread(os.path.join(y_big_dir, image_names[0]))

    if(len(ref_img.shape) == 2):
        gray = True
    else:
        gray = False
    
    # rescale images
    rescale_factor = 1./(2**bit_depth - 1)
    if(rescale_labels):
        rescale_factor_labels = rescale_factor
    else:
        rescale_factor_labels = 1
        
    while(True):
        
        if(gray):
            y_channels = 1
        else:
            y_channels = 3
            
        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, 1))
        y = np.zeros((batch_size, dim1, dim2, y_channels))
        
        # get one image at a time
        for i in range(batch_size):
                       
            # get random image
            img_index = np.random.randint(low=0, high=n_images)
            
            # open images
            x_big = skimage.io.imread(os.path.join(x_big_dir, image_names[img_index]))
            y_big = skimage.io.imread(os.path.join(y_big_dir, image_names[img_index]))

            # get random crop
            start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            patch_x = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] * rescale_factor
            patch_y = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] * rescale_factor_labels

            if(do_augmentation):
                
                rand_flip = np.random.randint(low=0, high=2)
                rand_rotate = np.random.randint(low=0, high=4)
                
                # flip
                if(rand_flip == 0):
                    patch_x = np.flip(patch_x, 0)
                    patch_y = np.flip(patch_y, 0)
                
                # rotate
                for rotate_index in range(rand_rotate):
                    patch_x = np.rot90(patch_x)
                    patch_y = np.rot90(patch_y)

                # illumination
                ifactor = 1 + np.random.uniform(-0.25, 0.25)
                patch_x *= ifactor
                    
            # save image to buffer
            x[i, :, :, 0] = patch_x
            
            if(gray):
                y[i, :, :, 0] = patch_y
            else:
                y[i, :, :, 0:y_channels] = patch_y
            
        # return the buffer
        yield(x, y)



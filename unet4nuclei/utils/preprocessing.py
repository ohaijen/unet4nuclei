import os

import numpy as np

import pathlib
from tqdm import tqdm

import skimage.io
import skimage.segmentation
import tensorflow as tf

def preprocess_input_images(raw_images_dir, normalized_images_dir):
    filelist = sorted(os.listdir(raw_images_dir))

    # run over all raw images
    for filename in tqdm(filelist):

        # load image and its annotation
        orig_img = skimage.io.imread(raw_images_dir + filename)

        # IMAGE

        # normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)

        img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later

        img = skimage.img_as_ubyte(img)
        img1 = img[:256,:256]
        skimage.io.imsave(normalized_images_dir + filename[:-4] + '_1.png', img1)
        img2 = img[256:512,:256]
        skimage.io.imsave(normalized_images_dir + filename[:-4] + '_2.png', img2)
        img3 = img[:256,256:512]
        skimage.io.imsave(normalized_images_dir + filename[:-4] + '_3.png', img3)
        img4 = img[256:512,256:512]
        skimage.io.imsave(normalized_images_dir + filename[:-4] + '_4.png', img4)

def preprocess_output_masks(raw_annotations_dir, boundary_labels_dir, boundary_size=2, min_nucleus_size=25):
    filelist = sorted(os.listdir(raw_annotations_dir))
    total_objects = 0

    # run over all raw images
    for filename in tqdm(filelist):

        # GET ANNOTATION
        annot = skimage.io.imread(raw_annotations_dir + filename)

        # strip the first channel
        if len(annot.shape) == 3:
            annot = annot[:,:,0]

        # label the annotations nicely to prepare for future filtering operation
        annot = skimage.morphology.label(annot)
        total_objects += len(np.unique(annot)) - 1

        # filter small objects, e.g. micronulcei
        annot = skimage.morphology.remove_small_objects(annot, min_size=min_nucleus_size)

        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot)

        for k in range(2, boundary_size, 2):
            boundaries = skimage.morphology.binary_dilation(boundaries)

        # BINARY LABEL

        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))

        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1

        # Split the image into four 256x256 byte squares, to make the UNet happy.
        # TODO(jen) - Pad the image to make a larger square instead?
        img1 = label_binary[:256,:256]
        skimage.io.imsave(boundary_labels_dir + filename[:-4] + '_1.png', img1)
        img2 = label_binary[256:512,:256]
        skimage.io.imsave(boundary_labels_dir + filename[:-4] + '_2.png', img2)
        img3 = label_binary[:256,256:512]
        skimage.io.imsave(boundary_labels_dir + filename[:-4] + '_3.png', img3)
        img4 = label_binary[256:512,256:512]
        skimage.io.imsave(boundary_labels_dir + filename[:-4] + '_4.png', img4)


    print("Total objects: ",total_objects)

def create_tf_examples(normalized_images_dir, boundary_labels_dir, tf_examples_dir):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    filelist = sorted(os.listdir(normalized_images_dir))


    # run over all raw images
    for filename in tqdm(filelist):
        with tf.python_io.TFRecordWriter(tf_examples_dir + filename[:-3] + "tfrecord") as writer:
            with open(normalized_images_dir + filename, mode='rb') as file: # b is important -> binary
                image = file.read()
            with open(boundary_labels_dir + filename, mode='rb') as file: # b is important -> binary
                mask = file.read()
            example = tf.train.Example(features = tf.train.Features(
            feature =
              {
            'image':_bytes_feature(image),
            'mask':_bytes_feature(mask)
               }))
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    datadir = "/Users/eiofinova/Documents/unet4nuclei_old/data/"
    raw_images_dir = os.path.join(datadir, "raw_images/")
    normalized_images_dir = os.path.join(datadir, "norm_images/")
    raw_annotations_dir = os.path.join(datadir, "raw_annotations/")
    boundary_labels_dir = os.path.join(datadir, "boundary_labels/")
    tf_examples_dir = os.path.join(datadir, "tf_examples/")

    preprocess_input_images(raw_images_dir, normalized_images_dir)
    preprocess_output_masks(raw_annotations_dir, boundary_labels_dir)
    create_tf_examples(normalized_images_dir, boundary_labels_dir, tf_examples_dir)


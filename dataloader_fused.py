from imutils import paths
import numpy as np
import argparse
import pickle
import random
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import keras

def show_image_triplet(a,p,n):
    """ this function shows the image triplet
    a  - ancor
    p - positive
    n - negative """
    fig = plt.figure(figsize=(8, 12))
    fig.add_subplot(3, 2, 1)
    plt.imshow(a[:,:,:3].astype('uint8'))
    plt.title("A image")
    fig.add_subplot(3, 2, 2)
    plt.title("A vector")
    plt.imshow(a[:,:,-1], cmap='gray', vmin=0, vmax=255)


    fig.add_subplot(3, 2, 3)
    plt.imshow(p[:,:,:3].astype('uint8'))
    plt.title("P image")
    fig.add_subplot(3, 2, 4)
    plt.title("P vector")
    plt.imshow(p[:,:,-1], cmap='gray', vmin=0, vmax=255)


    fig.add_subplot(3, 2, 5)
    plt.imshow(n[:,:,:3].astype('uint8'))
    plt.title("N image")
    fig.add_subplot(3, 2, 6)
    plt.title("N vector")
    plt.imshow(n[:,:,-1], cmap='gray', vmin=0, vmax=255)
    plt.suptitle("Ancor, Positive and Negative image")
    plt.show()


def show_image_fused(X):

    ''' the function takes an image with 4 channels and displays it '''
    fig = plt.figure(figsize=(8, 12))
    fig.add_subplot(1, 2, 1)
    plt.imshow(X[:,:,:3].astype('uint8'))
    plt.title("image")
    fig.add_subplot(1, 2, 2)
    plt.title("vector")
    plt.imshow(X[:,:,-1], cmap='gray', vmin=0, vmax=255)
    plt.suptitle("Image-Vector Pair")
    plt.show()


def process_image_pair(img, lbl,  n_channels_lbl):
    """ fucntion just create a single RGBI image from the initial image + semantic image"""
    image = load_img(img, target_size=(512, 512))
    lbl = load_img(lbl, target_size=(512, 512))
    image = img_to_array(image)
    lbl = np.rot90(lbl, k=1, axes=(1, 0))  # this line turns the image 90 clockwise, needed for IGN data

    # preprocess the image according to image net utils data
    # image = np.expand_dims(image, axis=0)
    #image = imagenet_utils.preprocess_input(image)

    # TODO: probably try to finish this function
    # preprocess the label image by creating an R,G,B, 1...N features image featuring
    # lbl_channels =  create_masks_from_rgb(lbl)

    if n_channels_lbl == 1:
    # pre-process image by just making it a grayscale image
        lbl = np.expand_dims((0.3 * lbl[:,:,0] + 0.59 * lbl[:,:,1] + 0.11 *  lbl[:,:,2]), axis=2)  # expand
    #else:
        #lbl = imagenet_utils.preprocess_input(image)
        #image = imagenet_utils.preprocess_input(image)
    # else keep color image for the moment
    # concatenate the resulting images horizontally
    image = np.dstack((image, lbl))
    #image = np.expand_dims(image, axis=0)
    return image


class fused_datagenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_im, batch_size=20, n_channels_img=3, n_channel_lbl=1):
        'Initialization'
        self.dataset_im = dataset_im
        self.batch_size = batch_size
        self.n_channels_img = n_channels_img
        self.n_channels_lbl = n_channel_lbl
        self.hard_mining_indexes = []
        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = list(range (index * self.batch_size, (index + 1) * self.batch_size))

        # Find list of IDs
        list_IDs_temp = [self.imagePaths[k] for k in indexes]

        # Generate data
        y = []
        X = []
        for i in range(0, len(list_IDs_temp)-1, 2):
            X.append(process_image_pair(list_IDs_temp[i], list_IDs_temp[i+1], n_channels_lbl = self.n_channels_lbl))
            y.append(list_IDs_temp[i])
        return  np.array(X), np.array(y)

    def getImagePaths(self):
        self.imagePaths = list(paths.list_images(self.dataset_im))
        self.imagePaths.sort()  # sort in alphabetical order
        self.total_images = len(self.imagePaths)
        print("There are %d images in this dataset ."%(len(self.imagePaths)))


    def addHardMiningIndexes(self, indexes):
        """ to add the hardMining indexes for the network training """
        self.hard_mining_indexes.vstack(indexes)

if __name__ == '__main__':
    topo_ortho_generator =  fused_datagenerator(dataset_im='/data/margokat/alegoria/processed_images/moselle')
    topo_ortho_generator.getImagePaths()
    X, path = topo_ortho_generator.__getitem__(11)






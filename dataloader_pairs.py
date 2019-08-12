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
    fig = plt.figure(figsize=(5, 8))
    fig.add_subplot(3, 2, 1)
    plt.imshow(a[:,:,:3])
    fig.add_subplot(3, 2, 2)
    plt.imshow(a[:,:,-1])


    fig.add_subplot(3, 2, 3)
    plt.imshow(p[:,:,:3])
    fig.add_subplot(3, 2, 4)
    plt.imshow(p[:,:,-1])

    fig.add_subplot(3, 2, 5)
    plt.imshow(n[:,:,:3])
    fig.add_subplot(3, 2, 6)
    plt.imshow(n[:,:,-1])
    plt.title("Ancor, Positive and Negative image")
    plt.show()


def process_image_pair(img, lbl):
    """ fucntion just create a single RGBI image from the initial image + semantic image"""
    image = load_img(img, target_size=(512, 512))
    lbl = load_img(lbl, target_size=(512, 512))
    image = img_to_array(image)
    lbl = np.rot90(lbl, k=1, axes=(1, 0))  # this line turns the image 90 clockwise, needed for IGN data

    # preprocess the image according to image net utils data
    # image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # preprocess the label image by creating an R,G,B, 1...N features image featuring
    # lbl_channels =  create_masks_from_rgb(lbl)

    # pre-process image by just making it a grayscale image
    gray = np.expand_dims(np.mean(lbl, -1), axis=2)  # expand

    # concatenate the resulting images horizontally
    image = np.dstack((image, gray))
    #image = np.expand_dims(image, axis=0)
    return image


class Hardmining_datagenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_2019, dataset_2004, batch_size=32, n_channels_img=3, n_channel_lbl=1):
        'Initialization'
        self.dataset_2019 = dataset_2019
        self.dataset_2004 = dataset_2004
        self.batch_size = batch_size
        self.n_channels_img = n_channels_img
        self.n_channels_lbl = n_channel_lbl
        self.hard_mining_indexes = []
        self.on_epoch_end()

    def getImagePaths(self):
        self.imagePaths2019 = list(paths.list_images(self.dataset_2019))
        self.imagePaths2019.sort()  # sort in alphabetical order


        self.imagePaths2004 = list(paths.list_images(self.dataset_2004))
        self.imagePaths2004.sort()  # sort in alphabetical order

        self.total_images = len(self.imagePaths2019)
        print("There are %d images in 2019 and %d in 2004."%(len(self.imagePaths2019),len(self.imagePaths2004)))


    def preComputePairsBatches(self, batch_number):
        """ initial computation, just return a batch of A, P, N pairs, batch_number - the corresponding positive image index"""

        # create initial positive and negative pairs: for each positive pair, make 500 random negative pairs
        # loop over the images
        batchImages = []
        batchPairs_indexes = []
        batchPairs_labels = []


        i = batch_number # the image to process now
        ancor = process_image_pair(self.imagePaths2019[i], self.imagePaths2019[i+1])
        positive = process_image_pair(self.imagePaths2004[i], self.imagePaths2004[i])
        # for each image
        for j in range(0, self.batch_size):
            # find batch non-corresponding images
            neg_index = np.random.randint(1,self.total_images-2)
            if neg_index == i: #if it is the same image that the positive one
                neg_index +=1
            if neg_index%2 == 0: #I want my negative label to be an image and not the label
                neg_index += 1
            negative = process_image_pair(self.imagePaths2019[neg_index], self.imagePaths2019[neg_index+1])

            # add pairs to the batch
            batchImages.append(np.stack([ancor, positive, negative]))
            batchPairs_indexes.append([i, i, neg_index])
            batchPairs_labels.append([1,0])

        return batchImages, batchPairs_indexes, batchPairs_labels

    def addHardMiningIndexes(self, indexes):
        """ to add the hardMining indexes for the network training """
        self.hard_mining_indexes.vstack(indexes)

if __name__ == '__main__':
    topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle', dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004')
    topo_ortho_generator.getImagePaths()

    for img in range(1):
        batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(img)
        # display an image just to check
        a,p,n =  batch_images[0]
        #show_image_triplet(a,p,n)




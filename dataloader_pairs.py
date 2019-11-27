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


def process_image_pair(img, lbl, n_channels_lbl, im_size):
    """ fucntion just create a single RGBI image from the initial image + semantic image"""
    image = load_img(img, target_size=im_size)
    lbl = load_img(lbl, target_size=im_size)
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
    else:
        lbl = imagenet_utils.preprocess_input(image)
        image = imagenet_utils.preprocess_input(image)
    # else keep color image for the moment
    # concatenate the resulting images horizontally
    image = np.dstack((image, lbl))
    #image = np.expand_dims(image, axis=0)
    return image


class Hardmining_datagenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_2019, dataset_2004, batch_size=4, n_channels_img=3, n_channel_lbl=1, im_size = (512,512), pairs = True):
        'Initialization'
        self.dataset_2019 = dataset_2019
        self.dataset_2004 = dataset_2004
        self.im_size = im_size
        self.batch_size = batch_size
        self.n_channels_img = n_channels_img
        self.n_channels_lbl = n_channel_lbl
        self.hard_mining_indexes = []
        self.hard_mining_images_ancors = []
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        self.pairs = pairs #or triplets

    def getImagePaths(self):
        self.imagePaths2019 = list(paths.list_images(self.dataset_2019))
        self.imagePaths2019.sort()  # sort in alphabetical order


        self.imagePaths2004 = list(paths.list_images(self.dataset_2004))
        self.imagePaths2004.sort()  # sort in alphabetical order

        self.total_images = int(len(self.imagePaths2019)/2)
        print("There are %d images in 2019 and %d in 2004."%(len(self.imagePaths2019)/2,len(self.imagePaths2004)/2))

    def createPairs(self):
        """ creates an 4d array with matches im 2019, label 2019 - im 2004, label 2004 (based on indexes)"""
        self.imagelookuptable = []
        for i in range(0, len(self.imagePaths2019),2):
            self.imagelookuptable.append([self.imagePaths2019[i], self.imagePaths2019[i+1],
                                    self.imagePaths2004[i], self.imagePaths2004[i+1]])



    def shuffleontheend(self):
        random.shuffle(self.imagelookuptable)


    def preComputePairsBatches(self, batch_number):
        """ initial computation, just return a batch of A, P, N pairs, batch_number - the corresponding positive image index"""

        # create initial positive and negative pairs: for each positive pair, make 500 random negative pairs
        # loop over the images
        batchImages = []
        batchPairs_indexes = []
        batchPairs_labels = []

        i = batch_number * self.batch_size  # the image to process now
        # for each image
        for j in range(0, self.batch_size):
            ancor = process_image_pair(self.imagelookuptable[i][0], self.imagelookuptable[i][1], self.n_channels_lbl,
                                       self.im_size)
            positive = process_image_pair(self.imagelookuptable[i][2], self.imagelookuptable[i][3], self.n_channels_lbl,
                                          self.im_size)
            # find batch non-corresponding images
            neg_index = np.random.randint(1,len(self.imagelookuptable))
            negative = process_image_pair(self.imagelookuptable[neg_index][0], self.imagelookuptable[neg_index][1],self.n_channels_lbl, self.im_size)

            # add pairs to the batch
            batchImages+=[[ancor, positive], [ancor, negative]]
            batchPairs_indexes+=[[i, i], [i, neg_index]]
            batchPairs_labels+=[1,0]
            i += 1 # update i

        return np.array(batchImages), np.array(batchPairs_indexes), np.array(batchPairs_labels)


    def preComputePairsBatchesHard(self, batch_number):
        """ initial computation, just return a batch of A, P, N pairs, batch_number - the corresponding positive image index"""

        # create initial positive and negative pairs where one is always a hard-sample

        batchImages = []
        batchPairs_indexes = []
        batchPairs_labels = []

        i = batch_number * self.batch_size  # the image to process now
        # for each image
        for j in range(0, self.batch_size):
            ancor = process_image_pair(self.imagelookuptable[i][0], self.imagelookuptable[i][1], self.n_channels_lbl,
                                       self.im_size)
            positive = process_image_pair(self.imagelookuptable[i][2], self.imagelookuptable[i][3], self.n_channels_lbl,
                                          self.im_size)
            # find batch non-corresponding images
            neg_index = np.random.randint(1,len(self.imagelookuptable))
            negative = process_image_pair(self.imagelookuptable[neg_index][0], self.imagelookuptable[neg_index][1],self.n_channels_lbl, self.im_size)

            # add pairs to the batch
            batchImages+=[[ancor, positive], [ancor, negative]]
            batchPairs_indexes+=[[i, i], [i, neg_index]]
            batchPairs_labels+=[1,0]
            i += 1 # update i
        # add one hard image
        
        X, y = self.__getitem__(batch_number)
        batchPairs_labels+=y
        batchImages+=X



        return np.array(batchImages), np.array(batchPairs_indexes), np.array(batchPairs_labels)







    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        if self.pairs: #case positive - negative image
            indexes = list(range (index * self.batch_size, (index + 1) * self.batch_size))

            # Find list of IDs
            try:
                list_anchors_temp = [self.hard_mining_images_ancors[k] for k in indexes]
                list_hard_temp = [self.hard_mining_samples_indexes[k] for k in indexes]
            except:
                index = random.randint(0, len(self.hard_mining_images_ancors)//self.batch_size - self.batch_size)
                indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))
                list_anchors_temp = [self.hard_mining_images_ancors[k] for k in indexes]
                list_hard_temp = [self.hard_mining_samples_indexes[k] for k in indexes]
            # Generate data
            y = []
            X = []
            for i in range(0, len(list_anchors_temp)):
                img, lbl = list_anchors_temp[i][0],list_anchors_temp[i][0].replace('img', 'lbl')
                ancor = process_image_pair(img, lbl, self.n_channels_lbl, self.im_size)
                img, lbl = list_anchors_temp[i][1],list_anchors_temp[i][1].replace('img', 'lbl')
                positive = process_image_pair(img, lbl,self.n_channels_lbl, self.im_size)
                random_hard = random.randint(0,len(list_hard_temp[i])-1)
                img, lbl = list_hard_temp[i][random_hard], list_hard_temp[i][random_hard].replace('img', 'lbl')
                negative = process_image_pair(img, lbl,self.n_channels_lbl, self.im_size)
                X += [[ancor, positive], [ancor, negative]]
                y += [1, 0]
            return X, y
        elif not self.pairs:
            indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))

            # Find list of IDs
            list_anchors_temp = [self.hard_mining_images_ancors[k] for k in indexes]
            list_hard_temp = [self.hard_mining_samples_indexes[k] for k in indexes]
            # Generate data
            X = []
            y = []
            for i in range(0, len(list_anchors_temp)):
                img, lbl = list_anchors_temp[i][0], list_anchors_temp[i][0].replace('img', 'lbl')
                ancor = process_image_pair(img, lbl, self.n_channels_lbl, self.im_size)
                img, lbl = list_anchors_temp[i][1], list_anchors_temp[i][1].replace('img', 'lbl')
                positive = process_image_pair(img, lbl, self.n_channels_lbl, self.im_size)
                random_hard = random.randint(0, len(list_hard_temp[i]) - 1)
                img, lbl = list_hard_temp[i][random_hard], list_hard_temp[i][random_hard].replace('img', 'lbl')
                negative = process_image_pair(img, lbl, self.n_channels_lbl, self.im_size)
                X += np.expand_dims(ancor, axis=0),  np.expand_dims(positive, axis=0), np.expand_dims(negative, axis=0)
                y += [1, 0]
                # .reshape(-1, self.im_size, self.im_size, self.n_channels_lbl)
            return [X[0],X[1],X[2]], None

    def addHardMiningIndexes(self, indexes):
        """ to add the hardMining indexes for the network training """
        self.hard_mining_samples_indexes = list(indexes.values())
        self.hard_mining_images_ancors = list(indexes.keys())
        assert len(self.hard_mining_samples_indexes) == len(self.hard_mining_images_ancors)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.hard_mining_indexes) / self.batch_size))

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

if __name__ == '__main__':
    topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle', dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004')
    topo_ortho_generator.getImagePaths()
    topo_ortho_generator.createPairs()
    for img in range(2):
        batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(img)
        # display an image just to check
        pos_idx = np.where(batchPairs_labels==1)
        neg_idx = np.where(batchPairs_labels==0)

        a = np.squeeze(batchPairs_images[pos_idx,0])
        p = np.squeeze(batchPairs_images[pos_idx,1])
        n = np.squeeze(batchPairs_images[neg_idx,1])
        labels = batchPairs_labels
       # print(a.shape)
        random_example = random.randint(0,topo_ortho_generator.batch_size)
        show_image_triplet(a[random_example],p[random_example],n[random_example])
        topo_ortho_generator.shuffleontheend()




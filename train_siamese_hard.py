from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator, show_image_triplet
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense,  Lambda, Activation, BatchNormalization
from sklearn.preprocessing import normalize
from tensorflow.keras.losses import cosine as cosineTF
import random
from keras.models import Model
from  keras.activations import relu
from keras import backend as K
# main file to train the siamese network
import numpy as np
from keras.callbacks import TensorBoard
from tensorboard_utils.tensorboard_utils import write_log, make_image
from optimizers.lookAhead import  Lookahead
from knn_distances_calculation import knn_distance_calculation

import tensorflow as tf
import os

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

def euclidean_distance(vects):
    x, y = vects
    # normalization layer ?
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    sum_square = K.sum( K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_distance(vects):
    ''' cosine distance implementation with normalization:
    reproduced from sklearn
	https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/pairwise.py#L655
    Cosine distance is defined as 1.0 minus the cosine similarity.

	'''
    x, y = vects
    # x = K.l2_normalize(x, axis=-1)
    # y = K.l2_normalize(y, axis=-1)

    # return K.dot(x,K.transpose(y)) #1 - is taken from distance cosine  sklearn (1- cos similarity)
    return 1.0 - cosineTF(x,y) # Cosine distance is defined as 1.0 minus the cosine similarity.


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    modified by myself to match sklearn impelemntation
    '''
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def contrastive_loss_per_pair(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return (y_true * sqaure_pred + (1 - y_true) * margin_square).eval()
def contrastive_loss_np(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
    sqaure_pred = np.square(y_pred)
    margin_square = np.square(np.maximum(margin - y_pred, 0))
    return np.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    '''
    y_pred = normalize(y_pred)
    pred = y_pred.ravel() <0.5
    return np.mean(pred == y_true)

def acc_keras(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    '''
    pred = K.tf.cast(K.less(y_pred,0.5), y_true.dtype)
    return K.mean(K.equal(K.flatten(y_true), K.flatten(pred)))

im_size = (512,512)
# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004',
                                                batch_size=4, n_channels_img=3, n_channel_lbl=1, im_size = im_size)
topo_ortho_generator.getImagePaths()
# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0,0].shape

# get the model
base_model = get_model(im_size[0], im_size[1], 4, pooling=True, weights='imagenet')
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# because we re-use the same instance `base_network = model`,
# the weights of the network
# will be shared across the two branches
processed_a = base_model(input_a)
processed_b = base_model(input_b)

# and then the regression based on two outputs (here subtraction)
distance = Lambda(euclidean_distance,
                  output_shape= eucl_dist_output_shape)([processed_a, processed_b])



model = Model([input_a, input_b], distance)
model.summary()
optimizer = Adam(lr=0.008, epsilon=None, decay =0.000000000001)
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics= [acc_keras])


tboard = TensorBoard(log_dir='./logs/hard_mining/', histogram_freq=0, write_graph=True)


for j in range(1):
    # num of epochs
    # HARD MINING and KNN-recalculation each round starts here
    hard_pairs = knn_distance_calculation(base_model,
                                          path_2019='/data/margokat/alegoria/processed_images/moselle/57-2015-0915-6895-LA93-0M50-E080',
                                          path_2004='/data/margokat/alegoria/processed_images/moselle_2004/57-2004-0915-6895-LA93-0M50-E080', bs =10)
    topo_ortho_generator.addHardMiningIndexes(hard_pairs)
    total_batches = len(hard_pairs)

    X, y  = topo_ortho_generator.__getitem__(0)
    pos_idx = 0
    neg_idx = 1
    a = np.squeeze(X[pos_idx, 0])
    p = np.squeeze(X[pos_idx, 1])
    n = np.squeeze(X[neg_idx, 1])
    labels = batchPairs_labels
    # print(a.shape)
    random_example = random.randint(0, topo_ortho_generator.batch_size)
    show_image_triplet(a[random_example], p[random_example], n[random_example])

    # #then train the model on the hard samples
    # for img in range(0, hard_pairs): # go th  topo_ortho_generator.total_images
    #
    #     model.fit_generator(generator=topo_ortho_generator,  steps_per_epoch= total_batches,
    #                 use_multiprocessing=True, callbacks = [tboard], workers=6)
    #
    #
    #     weights_conv = model.layers[1].get_weights()[0]
    #     print("re-trained weights")
    #     print(weights_conv)
    #     model.save_weights('models/siamese_bw_euclidean_' + str(j)+'_weights.h5')





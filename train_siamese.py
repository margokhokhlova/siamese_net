from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense,  Lambda, Activation, BatchNormalization

from tensorflow.keras.losses import cosine as cosineTF

from keras.models import Model
from keras import backend as K
# main file to train the siamese network
import numpy as np
from keras.callbacks import TensorBoard
from tensorboard_utils.tensorboard_utils import write_log
from optimizers.lookAhead import  Lookahead

import tensorflow as tf
import os

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
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

	'''
    x, y = vects
    # x = K.l2_normalize(x, axis=-1)
    # y = K.l2_normalize(y, axis=-1)

    # return K.dot(x,K.transpose(y)) #1 - is taken from distance cosine correlation sklearn
    return cosineTF(x,y)


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    modified by myself to match sklearn impelemntation
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return  K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
def contrastive_loss_per_pair(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return (y_true * sqaure_pred + (1 - y_true) * margin_square).eval()


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    '''
    pred = y_pred.ravel() <0.5
    return np.mean(pred == y_true)

def acc_keras(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    '''
    pred = K.cast(y_pred < 0.5,dtype='float32')
    return K.mean(K.equal(y_true, pred))

# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004',n_channels_img=3, n_channel_lbl=1)
topo_ortho_generator.getImagePaths()
# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0,0].shape

# get the model
base_model = get_model(512, 512, 4, pooling=True, weights='imagenet')
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# because we re-use the same instance `base_network = model`,
# the weights of the network
# will be shared across the two branches
processed_a = base_model(input_a)
processed_b = base_model(input_b)

# and then the regression based on two outputs (here subtraction)
distance = Lambda(cosine_distance,
                  output_shape= eucl_dist_output_shape)([processed_a, processed_b])



model = Model([input_a, input_b], distance)
model.summary()
optimizer = Adam(lr=0.005, beta_1=0.95, beta_2=0.989, epsilon=None, decay=0.00000001, amsgrad=True)
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics= [acc_keras])

lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model) # add into model

#tboard = TensorBoard(log_dir='logs/', histogram_freq=0,
#          write_graph=True)

log_path = './logs'
callback = TensorBoard(log_path)
callback.set_model(model)
train_names = ['train_loss cosine', 'acc with threshold 0.5 cosine']

for j in range(10): # num of epochs
    for img in range(0, 12000, 2): # go th  topo_ortho_generator.total_images
        batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(img)
        # before each new epoch, do the hard mining
        y_pred = model.predict_on_batch(
            [batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
        #test_loss = contrastive_loss_per_pair(batchPairs_labels,y_pred)
        # TODO: add the batch modification to do hard mining
        logs = model.train_on_batch([batchPairs_images[:,0], batchPairs_images[:,1]], batchPairs_labels)
        write_log(callback, train_names, logs,  img/2 + j*6000)
        # compute final accuracy on the training  set
        if img%3000==0:
            y_pred = model.predict_on_batch([batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
            tr_acc = compute_accuracy(batchPairs_labels, y_pred)
            print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    model.save_weights('models/siamese_bw_cosine' + str(j)+'_weights.h5')




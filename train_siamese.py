from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator
from keras.optimizers import Adam,SGD
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense,  Lambda, Activation, BatchNormalization
from sklearn.preprocessing import normalize
from tensorflow.keras.losses import cosine as cosineTF
from keras.utils import multi_gpu_model
from keras.models import Model
from keras import backend as K
# main file to train the siamese network
import numpy as np
from keras.callbacks import TensorBoard
from tensorboard_utils.tensorboard_utils import write_log, make_image
from optimizers.lookAhead import  Lookahead
from knn_distances_calculation import knn_distance_calculation

import tensorflow as tf
import os

# #The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def euclidean_distance(vects):
    x, y = vects

    sum_square = K.sum( K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_distance(vects):
    ''' cosine distance implementation with normalization:
    reproduced from sklearn
	https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/pairwise.py#L655
    Cosine distance is defined as 1.0 minus the cosine similarity.

	'''
    x, y = vects
    # return K.dot(x,K.transpose(y)) #1 - is taken from distance cosine  sklearn (1- cos similarity)
    return 1- cosineTF(x,y) # Cosine distance is defined as 1.0 minus the cosine similarity.


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred, margin = 1):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# def contrastive_loss(y_true, y_pred, margin = 1):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
#     y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
#     #square_pred = K.square(y_pred)
#     margin_square =  K.maximum(margin - y_pred, 0) #K.square(
#     return K.mean(y_true * y_pred + (1 - y_true) * margin_square)

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
    pred = y_pred.ravel() <0.5
    return np.mean(pred == y_true)

def acc_keras(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    '''
    pred = K.tf.cast(K.less(y_pred,0.5), y_true.dtype)
    return K.mean(K.equal(K.flatten(y_true), K.flatten(pred)))

im_size = (256,256)
# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004',
                                                batch_size=13, n_channels_img=3, n_channel_lbl=1, im_size = im_size)
topo_ortho_generator.getImagePaths()
# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0,0].shape

# get the model
base_model = get_model(im_size[0], im_size[1], 4, pooling=False, weights='imagenet')

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
optimizer = SGD(lr=0.0001, clipvalue=1.0)
try:
    model = multi_gpu_model(model,  gpus=2)
except:
    print('multi-gpu failed, using a single gpu')
    pass


model.compile(loss=contrastive_loss, optimizer=optimizer, metrics= [acc_keras])

# lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
# lookahead.inject(model) # add into model

#tboard = TensorBoard(log_dir='logs/', histogram_freq=0,
#          write_graph=True)

log_path = './logs/siamese_nonhard'
callback = TensorBoard(log_path)
callback.set_model(model)
train_names = ['train_loss cosine', 'acc with threshold 0.5 cosine']
epoch_logs = [0,0]
av_counter = 0
# _, knn_res = knn_distance_calculation(base_model,
#                                       path_2019='/data/margokat/alegoria/processed_images/moselle/',
#                                       path_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
#                                       bs=10)
for j in range(0, 20): # num of epochs
    _, knn_res = knn_distance_calculation(base_model,
                                          path_2019='/data/margokat/alegoria/processed_images/moselle/',
                                          path_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                          bs=20, feat_shape=128, im_size = im_size)

    write_log(callback, ['map@5'], [knn_res], j)
    for img in range(0, 12000, 2): # go th  topo_ortho_generator.total_images
        batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(img)
        #batchPairs_images[0, 1]=  batchPairs_images[0, 0] # add same images from the same year
        # #before each new epoch, do the hard mining
        # y_pred = model.predict_on_batch(
        #      [batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
        # test_loss = contrastive_loss_np(batchPairs_labels,y_pred)
        # # TODO: add the batch modification to do hard mining
        logs = model.train_on_batch([batchPairs_images[:,0], batchPairs_images[:,1]], batchPairs_labels)
        epoch_logs[0] += logs[0]
        epoch_logs[1] += logs[1]
        # compute final accuracy on the training  set
        if img%1000==0:
            write_log(callback, ['train_loss average euclidean', ' average acc with threshold 0.5 euclidean'],
                      map(lambda x: x / 500, epoch_logs), av_counter)
            av_counter+=1
            epoch_logs = [0, 0]

            # y_pred = model.predict_on_batch([batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
            # tr_acc = compute_accuracy(batchPairs_labels, y_pred)
            # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))


    base_model.save_weights('models/base_model_siamese_no_hard_' + str(j)+'_weights.h5')





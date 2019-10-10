from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator, show_image_triplet
from keras.optimizers import Adam
from keras.layers import Input, Dense,  Lambda, Activation, BatchNormalization, Layer
from tensorflow.keras.losses import cosine as cosineTF
from keras.models import Model
from keras import backend as K
# main file to train the siamese network
import numpy as np
from keras.callbacks import TensorBoard
from tensorboard_utils.tensorboard_utils import  TensorBoardWrapper
from optimizers.lookAhead import  Lookahead
from knn_distances_calculation import knn_distance_calculation
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import os

# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";
# config = tf.ConfigProto()
# session = tf.Session(config=config)
# K.set_session(session)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum( K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def cosine_distance(vects):
    ''' cosine distance :
    reproduced from sklearn
	https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/pairwise.py#L655
    Cosine distance is defined as 1.0 minus the cosine similarity.
	'''
    x, y = vects
    return 1.0 - cosineTF(x,y)



# alternative loss
# https://github.com/david-husx/crossview_localisation/blob/master/src/CVM-Net/train.py

#inspired by https://nbviewer.jupyter.org/github/krasserm/face-recognition/blob/master/face-recognition.ipynb?flush_cache=true

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

def contrastive_loss_np(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    this is a numpy version for debugging
    '''
    margin = 1
    sqaure_pred = np.square(y_pred)
    margin_square = np.square(np.maximum(margin - y_pred, 0))
    return np.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    numpy debug version
    '''
    pred = y_pred.ravel() <0.5
    return np.mean(pred == y_true)

def acc_keras(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    So far the threshold is 0.5, to be estimated correctly
    Tensor-based version
    '''
    pred = K.tf.cast(K.less(y_pred,0.1), y_true.dtype)
    return K.mean(K.equal(K.flatten(y_true), K.flatten(pred)))


ALPHA = 0.1  # Triplet Loss Parameter
def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss




im_size = (512,512) # setup for the image size
# get the data training data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                batch_size=1, n_channels_img=3, n_channel_lbl=1, im_size = im_size, pairs =False)
topo_ortho_generator.getImagePaths()
# get the data validation data
topo_ortho_generator_valid = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/basrhin_2019/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/basrhin_2004/',
                                                batch_size=10, n_channels_img=3, n_channel_lbl=1, im_size = im_size)
topo_ortho_generator_valid.getImagePaths()

# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0,0].shape

# get the base model (back-bone)
base_model = get_model(im_size[0], im_size[1], 4, pooling=False, weights='imagenet') # version with or without pooling

# because we re-use the same instance `base_network = model`,
# the weights of the network
# will be shared across the two branches
input_a = Input(shape=input_shape,name="anchor_input")
input_p = Input(shape=input_shape, name="positive_input")
input_n = Input(shape=input_shape, name="negative_input")

anchor_input = base_model(input_a)
positive_input = base_model(input_p)
negative_input = base_model(input_n)


# and then the regression based on two outputs (here subtraction)


# TripletLoss Layer
loss_layer = TripletLossLayer(alpha=0.15, name='triplet_loss_layer')([anchor_input, positive_input, negative_input])

# Connect the inputs with the outputs
model = Model(inputs=[input_a, input_p, input_n], outputs=loss_layer)

model.summary()
optimizer = Adam(lr=0.0001, epsilon=None)
model.compile(optimizer=optimizer, metrics= [acc_keras])

tb_cb = TensorBoard(log_dir='./logs/euclidean_hard_triplet', write_images=1, histogram_freq=0)

epochs_per_rounds =30
knn_res = np.zeros(epochs_per_rounds)
knn_res_validation = np.zeros(2)
# calculate the hard pairs before starting to train the net
# hard_pairs_valid, knn_res_validation[0] = knn_distance_calculation(base_model,
#                                                   path_2019='/data/margokat/alegoria/processed_images/basrhin_2019/',
#                                                   path_2004='/data/margokat/alegoria/processed_images/basrhin_2004/',
#                                                   bs=10)
# topo_ortho_generator_valid.addHardMiningIndexes(hard_pairs_valid)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)
for j in range(3):
    # num of epochs
    # HARD MINING and KNN-recalculation each round starts here
    hard_pairs, knn_res[j] = knn_distance_calculation(base_model,
                                          path_2019='/data/margokat/alegoria/processed_images/moselle/',
                                          path_2004='/data/margokat/alegoria/processed_images/moselle_2004/', bs =10)
    topo_ortho_generator.addHardMiningIndexes(hard_pairs)
    total_batches = len(hard_pairs)
    # X = topo_ortho_generator.__getitem__(0)
    # #then train the model on the hard samples
    # logs = model.train_on_batch(X)
    model.fit_generator(generator=topo_ortho_generator,
                        epochs =epochs_per_rounds + j*epochs_per_rounds,
                        steps_per_epoch= int(total_batches/topo_ortho_generator.batch_size),
                        # validation_data = topo_ortho_generator_valid,
                        # validation_steps = int(len(hard_pairs_valid)/topo_ortho_generator_valid.batch_size),
                        callbacks=[tb_cb, reduce_lr],
                use_multiprocessing=True, initial_epoch=j*epochs_per_rounds)


    # model.save_weights('models/siamese_bw_euclidean_hardmining_triplet' + str(j)+'_weights.h5')


# after all the training is done, re-calculate the KNN results
# _, knn_res_validation[2] = knn_distance_calculation(base_model,
#                                       path_2019='/data/margokat/alegoria/processed_images/basrhin_2019/',
#                                       path_2004='/data/margokat/alegoria/processed_images/basrhin_2004/', bs =10)
print('Knn MAP training')
print(knn_res)
print("Validation knn map before/after training")
print(knn_res_validation)
model.save("'models/triplet_model.h5")
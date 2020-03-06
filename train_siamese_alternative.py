from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator, show_image_triplet
from keras.optimizers import Adam, SGD
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
from knn_distances_calculation import knn_distance_calculation
import tensorflow as tf
import os
from tensorboard_utils.tensorboard_utils import write_log, make_image
from focal_loss import focal_loss

def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

#  inspiration https://github.com/sorenbouma/keras-oneshot/blob/master/SiameseNet.ipynb
im_size = (256,256)
# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/meurthe_2019/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/meurthe_2004/',
                                                batch_size=6, n_channels_img=3, n_channel_lbl=1, im_size = im_size, augmentation=True)
topo_ortho_generator.getImagePaths()
topo_ortho_generator.createPairs()
topo_ortho_generator_valid = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                batch_size=10, n_channels_img=3, n_channel_lbl=1, im_size = im_size, augmentation=True)



# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0, 0].shape

# get the base model (back-bone)
base_model = get_model(im_size[0], im_size[1], 4, pooling=False, weights='imagenet')  # version with or without pooling
# because we re-use the same instance `base_network = model`,
# the weights of the network
# will be shared across the two branches
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_model(input_a)
processed_b = base_model(input_b)

# and then the regression based on two outputs (here subtraction)
L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1])) #Lambda(lambda tensors: K.dot(tensors[0],tensors[1]))
#TODO: https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/ Here is the simple algorithm to extend SIFT to RootSIFT:
#TODO:  In the past, the 2:4:6 rule (negative powers of 10) has worked quite well for me — using a learning rate of 10^-6 for the bottommost few layers, 10^-4 for the other transfer layers and 10^-2 for any additional layers we added.
L1_distance = L1_layer([processed_a, processed_b])
prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)
model = Model(inputs=[input_a, input_b], outputs=prediction)
#     model = multi_gpu_model(model)
# except:
#     pass

optimizer = Adam(0.00005)
# //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

log_path = './logs/cosine_hard_prediction/bw_label_128_crossval'
callback = TensorBoard(log_path)
callback.set_model(model)
epoch_logs = [0, 0]
av_counter = 0

for j in range(0, 100):  # num of epochs
    if j % 5 == 0:

        hard_pairs, knn_res = knn_distance_calculation(base_model,
                                                       path_2019='/data/margokat/alegoria/processed_images/meurthe_2019/',
                                                       path_2004='/data/margokat/alegoria/processed_images/meurthe_2004/',
                                                       bs=10, feat_shape=128, im_size=im_size,n_channel_lbl=1)
        write_log(callback, ['map@5'], [knn_res], j)
        topo_ortho_generator.addHardMiningIndexes(hard_pairs)
        _, knn_res_val = knn_distance_calculation(base_model,
                                                       path_2019='/data/margokat/alegoria/processed_images/moselle/',
                                                       path_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                       bs=10, feat_shape=128, im_size=im_size,n_channel_lbl=1)
        write_log(callback, ['map@5_val'], [knn_res_val], j)
        _, knn_res_test = knn_distance_calculation(base_model,
                                                          path_2019='/data/margokat/alegoria/processed_images/basrhin_2019/',
                                                          path_2004='/data/margokat/alegoria/processed_images/basrhin_2004/',
                                                          bs=10, feat_shape = 128, im_size = im_size, n_channel_lbl=1)

        write_log(callback, ['map@5_test'], [knn_res_test], j)
    for img in range(0, int(
            topo_ortho_generator.total_images / topo_ortho_generator.batch_size)):  # go th  topo_ortho_generator.total_images
        batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatchesHard(img)
        # batchPairs_images[0, 1]=  batchPairs_images[0, 0] # add same images from the same year
        # #before each new epoch, do the hard mining
        # y_pred = model.predict_on_batch(
        #      [batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
        # test_loss = contrastive_loss_np(batchPairs_labels,y_pred)
        # # TODO: add the batch modification to do hard mining
        logs = model.train_on_batch([batchPairs_images[:, 0], batchPairs_images[:, 1]], batchPairs_labels)
        epoch_logs[0] += logs[0]
        epoch_logs[1] += logs[1]
        # compute final accuracy on the training  set

    write_log(callback, ['train_loss average cosine', ' average acc with threshold 0.5 cosine'],
              map(lambda x: x / img, epoch_logs), j)
    epoch_logs = [0, 0]

        # y_pred = model.predict_on_batch([batchPairs_images[:, 0], batchPairs_images[:, 1]])  # temp solution, tp check later on
        # tr_acc = compute_accuracy(batchPairs_labels, y_pred)
        # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    if j>0 and j%5 ==0:
        base_model.save_weights('models/base_model_siamese_prediction_bw_label_128_crossval_' + str(j) + '_weights.h5')
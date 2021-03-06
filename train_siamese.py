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
from keras_lr_multiplier import LRMultiplier
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


im_size = (256,256)
# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019= '/data/margokat/alegoria/processed_images/meurthe_2019',
                                                dataset_2004='/data/margokat/alegoria/processed_images/meurthe_2004',
                                                batch_size=6, n_channels_img=3, n_channel_lbl=3, im_size = im_size, augmentation=True)
topo_ortho_generator.getImagePaths()
topo_ortho_generator.createPairs()
topo_ortho_generator_valid = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                batch_size=6, n_channels_img=3, n_channel_lbl=3, im_size = im_size, augmentation=True)
topo_ortho_generator_valid.getImagePaths()
topo_ortho_generator_valid.createPairs()


# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0, 0].shape

# get the base model (back-bone)
base_model = get_model(im_size[0], im_size[1], 6, pooling=False, weights='imagenet', final_dimension = 128)  # version with or without pooling

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

L1_distance = L1_layer([processed_a, processed_b])
prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)
model = Model(inputs=[input_a, input_b], outputs=prediction)

optimizer = optimizer = Adam(lr=0.0001, decay = 0.0000001)#LRMultiplier('adam', {'fusion': 0.0001, 'resnet50': 0.00005, 'conv_margo_1': 0.001, 'conv_margo_2':0.001, 'conv_margo_3':0.001})
# //TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

log_path = './logs/cosine_hard_prediction/label_128_batchnorm_rgb_main'
callback = TensorBoard(log_path)
callback.set_model(model)
epoch_logs = [0, 0]
av_counter = 0

#main training loop
for j in range(0, 120):  # num of epochs
    if j % 5 == 0:
        hard_pairs, knn_res = knn_distance_calculation(base_model,
                                                 path_2019='/data/margokat/alegoria/processed_images/moselle',
                                                 path_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                 bs=10, feat_shape=128, im_size=im_size,n_channel_lbl=3)
        write_log(callback, ['map@5'], [knn_res], j)

        topo_ortho_generator.addHardMiningIndexes(hard_pairs)

        _, knn_res_validation = knn_distance_calculation(base_model,
                                                          path_2019='/data/margokat/alegoria/processed_images/basrhin_2019/',
                                                          path_2004='/data/margokat/alegoria/processed_images/basrhin_2004/',
                                                          bs=10, feat_shape = 128, im_size = im_size, n_channel_lbl=3)


        
        write_log(callback, ['map@5_val'], [knn_res_validation], j)



        _, knn_res_test = knn_distance_calculation(base_model,
                                                          path_2019='/data/margokat/alegoria/processed_images/meurthe_2019/',
                                                          path_2004='/data/margokat/alegoria/processed_images/meurthe_2004/',

                                                       bs=10, feat_shape=128, im_size=im_size, n_channel_lbl=3)
       
        write_log(callback, ['map@5_test'], [knn_res_test], j)

    for img in range(0, int(
            topo_ortho_generator.total_images / topo_ortho_generator.batch_size)):  # go th  topo_ortho_generator.total_images
        batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatchesHard(img)
        logs = model.train_on_batch([batchPairs_images[:, 0], batchPairs_images[:, 1]], batchPairs_labels)
        epoch_logs[0] += logs[0]
        epoch_logs[1] += logs[1]
        # compute final accuracy on the training  set

    write_log(callback, ['train_loss average', ' accuracy average'],
              map(lambda x: x / img, epoch_logs), j)
    epoch_logs = [0, 0]

    # the model is saved each 10 epochs
    if j>0 and j%10 ==0:
        base_model.save_weights('models/label_128_rgb_batchnorm_bce_main' + str(j) + '_weights.h5')
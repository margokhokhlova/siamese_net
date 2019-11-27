from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator, show_image_triplet
from keras.optimizers import Adam, SGD
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from tensorflow.keras.losses import cosine as cosineTF
from keras.models import Model
from keras import backend as K
# main file to train the siamese network
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
from knn_distances_calculation import knn_distance_calculation
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import os
from tensorboard_utils.tensorboard_utils import write_log, make_image

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=np.random.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
config = tf.ConfigProto()
session = tf.Session(config=config)
K.set_session(session)

#  inspiration https://github.com/sorenbouma/keras-oneshot/blob/master/SiameseNet.ipynb

im_size = (512,512) # setup for the image size
# get the data training data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle/',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004/',
                                                batch_size=3, n_channels_img=3, n_channel_lbl=1, im_size = im_size)
topo_ortho_generator.getImagePaths()
topo_ortho_generator.createPairs()
# get one test batch
batch_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(0)
input_shape = batch_images[0,0].shape

# get the base model (back-bone)
base_model = get_model(im_size[0], im_size[1], 4, pooling=False, weights='imagenet') # version with or without pooling
base_model.load_weights('/home/margokat/projects/siamese_net/models/base_model_siamese_no_hard_9_weights.h5', by_name=True)
# because we re-use the same instance `base_network = model`,
# the weights of the network
# will be shared across the two branches
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_model(input_a)
processed_b = base_model(input_b)

# and then the regression based on two outputs (here subtraction)
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([processed_a, processed_b])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
model = Model(inputs=[input_a,input_b],outputs=prediction)
optimizer = SGD(lr=0.00001, clipvalue=1.0)
# try:
#     model = multi_gpu_model(model)
# except:
#     pass

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
model.compile(loss="binary_crossentropy",optimizer=optimizer)

tb_cb = TensorBoard(log_dir='./logs/cosine_hard_pretrained', write_images=1, histogram_freq=0)

epochs_per_rounds =1
knn_res = np.zeros(epochs_per_rounds)
knn_res_validation = np.zeros(2)
#calculate the hard pairs before starting to train the net
# hard_pairs_valid, knn_res_validation[0] = knn_distance_calculation(base_model,
#                                                   path_2019='/data/margokat/alegoria/processed_images/basrhin_2019',
#                                                   path_2004='/data/margokat/alegoria/processed_images/basrhin_2004',
#                                                   bs=10, feat_shape = 512, im_size = im_size)
# topo_ortho_generator_valid.addHardMiningIndexes(hard_pairs_valid)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.000001)
log_path = './logs/cosine_hard_pretrained'
callback = TensorBoard(log_path)
callback.set_model(model)

for j in range(15):
    # num of epochs
    # HARD MINING and KNN-recalculation each round starts here
    hard_pairs, knn_res[j] = knn_distance_calculation(base_model,
                                          path_2019='/data/margokat/alegoria/processed_images/moselle',
                                          path_2004='/data/margokat/alegoria/processed_images/moselle_2004', bs =5, feat_shape =512, im_size = im_size)
    write_log(callback, ['map@5'], [knn_res[j]], j)
    topo_ortho_generator.addHardMiningIndexes(hard_pairs)
    total_batches = len(hard_pairs)
    print('total batches ', total_batches)
    x, y  = topo_ortho_generator.__getitem__()
    #then train the model on the hard samples
    model.fit_generator(generator=topo_ortho_generator,
                        epochs =epochs_per_rounds + j*epochs_per_rounds,
                        steps_per_epoch= int(total_batches/topo_ortho_generator.batch_size),
                        # validation_data = topo_ortho_generator_valid,
                        # validation_steps = int(len(hard_pairs_valid)/topo_ortho_generator_valid.batch_size),
                        callbacks=[tb_cb, reduce_lr],
                        use_multiprocessing=True, initial_epoch=j*epochs_per_rounds)


    base_model.save_weights('models/siamese_base_cosine_hard_nopulling' + str(j)+'_weights.h5')



model.save("'models/final_model_nopulling_hard.h5")
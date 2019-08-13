from model_for_siamese import get_model
from dataloader_pairs import Hardmining_datagenerator
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers import Input, Dense,  Lambda, Activation
from keras.models import Model
from keras import backend as K
# main file to train the siamese network


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)



# get the data
topo_ortho_generator = Hardmining_datagenerator(dataset_2019='/data/margokat/alegoria/processed_images/moselle',
                                                dataset_2004='/data/margokat/alegoria/processed_images/moselle_2004')
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
processed_a = Activation('relu')(base_model(input_a))
processed_b = Activation('relu') (base_model(input_b))

# and then the regression based on two outputs (here subtraction)
distance = Lambda(euclidean_distance,
                  output_shape= eucl_dist_output_shape)([processed_a, processed_b])



model = Model([input_a, input_b], distance)
model.summary()
optimizer = Adam(lr=0.005, beta_1=0.95, beta_2=0.989, epsilon=None, decay=0.00000001, amsgrad=True)
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['acc'])

for img in range(topo_ortho_generator.total_images): # go th
    batchPairs_images, batchPairs_indexes, batchPairs_labels = topo_ortho_generator.preComputePairsBatches(img)
    # before each new epoch, do the hard mining
    y_pred = model.predict_on_batch([batchPairs_images[:,0], batchPairs_images[:,1]])
    print(y_pred.shape)
    print(batchPairs_labels.shape)
    test_loss = contrastive_loss(batchPairs_labels, y_pred)



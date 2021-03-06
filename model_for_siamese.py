from keras.applications import ResNet50
import numpy as np
from keras.layers import GlobalMaxPooling2D, Dense, Conv2D, Lambda, Activation, BatchNormalization, Dropout, Flatten
from keras.models import Model,  Sequential, Input
from keras.applications import imagenet_utils
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU


def get_model (H,W,C, pooling =True, weights = 'imagenet', fusion = 'conv',final_dimension = 128):
    ''' Loads the model, the core is Resnet50, the output of the last covolutional layer is the output,
    either with or without maxpooling, pre-trained or not.
    Parameters:
        H, W - input dimensionality
        C - number of color channels
        pooling - with or witout pooling
        weights - 'imagenet' or None
        fusion - currently, only fusion model for multi-dim features
        final_dimensions - the final dimensionality of the descriptor
        returns = non-compiled Keras Model
    '''

    # load the ResNet50 network and store the batch size in a convenience
    # variable
    print("[INFO] loading network...")
    model_top = Sequential()

    if fusion != 'conv':
        model_top.add(Dense(3, activation=None, input_shape=(H,W,C)))
    elif fusion == 'conv':
        model_top.add(Conv2D(filters=3, kernel_size=(1,1), strides=(1, 1), activation = None, padding = 'same', input_shape=(H,W,C), name ='fusion'))
        model_top.add(Lambda(imagenet_utils.preprocess_input)) # custom pre-processing layer
        if C==4:
            w = np.array([[[[1 / 2, 0, 0],
                            [0, 1 / 2, 0],
                            [0, 0, 1 / 2],
                            [1 / 2, 1 / 2, 1 / 2]]]], dtype=float), np.array([0., 0., 0.], dtype=float)
        elif C==6:
            w = np.array([[[[0.5, 0., 0.],
            [0., 0.5, 0.],
            [0., 0., 0.5],
            [0.5, 0., 0.],
            [0., 0.5, 0.],
            [0., 0., 0.5]]]], dtype=float), np.array([0., 0., 0.], dtype=float)

        model_top.layers[0].set_weights(w)
    baseModel = ResNet50(weights=weights, include_top=False)
    model_top.add(baseModel)
    if pooling:
        # add the global max pooling layer
        headModel = model_top.output
        headModel = GlobalMaxPooling2D()(headModel)  # add a GLOBAL MAXPOOLING layer to obtain one value per channel
        model = Model(inputs = model_top.input, outputs=headModel)
        shape_feat =  2048

    else:
        # adding more layers to the original network
        # 16,16,2048
        # Recommended > CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        if H == 512:
            model = model_top
            model_top.add(Conv2D(filters=1024, kernel_size = (3,3), padding='valid', strides =2, kernel_initializer='he_normal', name="conv_margo_1",kernel_regularizer=l2(2e-4)))
            model.add(Activation('tanh'))
            # model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_1'))
            # now the features should be  7,7,1024
            model_top.add(Conv2D(filters=512, kernel_size=(3, 3),padding='valid', kernel_initializer='he_normal',
                                 name="conv_margo_2",kernel_regularizer=l2(2e-4)))
            model.add(Activation('tanh'))
            # model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_2'))
            #   now the features should be 5x5x512
            model_top.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                 name="conv_margo_3", kernel_regularizer=l2(2e-4)))
            model.add(Activation('tanh'))
            # model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_3'))
            #   now the features should be 5x5x128
            # add a FC layer + softmax or sigmoid activation

            shape_feat = 5 * 5 * 256
            model.add(Flatten())
            model.add(Dense(512, input_shape=(shape_feat,), kernel_regularizer=l2(1e-3)))
        elif H == 256:
            #8x8x16
            model = model_top
            model_top.add(
                Conv2D(filters=1024, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal',
                       name="conv_margo_1"))
            model.add(Activation('tanh'))
            model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_1'))
            # now the features should be  6,6,1024
            model_top.add(Conv2D(filters=512, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal',
                                 name="conv_margo_2"))
            model.add(Activation('tanh'))
            model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_2'))
            # model_top.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
            #                      name="conv_margo_3"))
            # model.add(Activation('tanh'))
            #model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_3'))
            shape_feat = 4 * 4 * 512
            model.add(Flatten())
            model.add(Dense(final_dimension, input_shape=(shape_feat,)))


    model.summary()

    for l in model.layers:

        if l.name == 'fusion':
            l.trainable = True
        # elif l.name == 'resnet50':
        #      l.trainable = False  # uncomment if only train the new layers
        print(l.name, l.trainable)


    return model









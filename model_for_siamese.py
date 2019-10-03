from keras.applications import ResNet50
import numpy as np
from keras.layers import GlobalMaxPooling2D, Dense, Conv2D, Lambda, Activation, BatchNormalization, Dropout, Flatten
from keras.models import Model,  Sequential, Input
from keras.applications import imagenet_utils
from keras.layers.advanced_activations import LeakyReLU
import argparse

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-imsize", "--img-size", default = 512, help="Input Image dimensions")
# ap.add_argument("-c", "--channels", type=int, default=3,
#     help="number of channels in the image: RGB = 3, RGBI = 4, etc")
# ap.add_argument("-p", "--pooling", type=bool, default=True,
#     help="with max pool or no?")
# args = vars(ap.parse_args())
# H = W = args['img_size']
# C = args['channels']
# pooling = args["pooling"]



def get_model (H,W,C, pooling =True, weights = 'imagenet', fusion = 'conv'):
    ''' Loads the model, the core is Resnet50, the output of the last covolutional layer is the output,
    either with or without maxpooling, pre-trained or not.
    Parameters:
        H, W - input dimensionality
        C - number of color channels
        pooling - with or witout pooling
        weights - 'imagenet' or None
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
        w = np.array([[[[1 / 2, 0, 0],
                        [0, 1 / 2, 0],
                        [0, 0, 1 / 2],
                        [1 / 2, 1 / 2, 1 / 2]]]], dtype=float), np.array([0., 0., 0.], dtype=float)

        model_top.layers[0].set_weights(w)
    baseModel = ResNet50(weights=weights, include_top=False)
    model_top.add(baseModel)
    # add the global max pooling layer
    headModel = model_top.output
    headModel = GlobalMaxPooling2D()(headModel) # add a GLOBAL MAXPOOLING layer to obtain one value per channel
    if pooling:
        model = Model(inputs = model_top.input, outputs=headModel)
        shape_feat =  2048

    else:
        # adding more layers to the original network
        # 16,16,2048
        # Recommended > CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        model = model_top
        model_top.add(Conv2D(filters=1024, kernel_size = (3,3), padding='valid', strides =2, kernel_initializer='he_normal', name="conv_margo_1"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_1'))
        # now the features should be  7,7,1024
        model_top.add(Conv2D(filters=512, kernel_size=(3, 3),padding='valid', kernel_initializer='he_normal',
                             name="conv_margo_2"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_2'))
        #   now the features should be 5x5x512
        model_top.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                             name="conv_margo_3"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=3, name='Batch_norm_Margo_3'))
        #   now the features should be 5x5x128
        # add a FC layer + softmax activation

        shape_feat = 5 * 5 * 256
        model.add(Flatten())
        model.add(Dense(512, input_shape=(shape_feat,), activation="softmax"))

    model.summary()

    for l in model.layers:

        if l.name == 'fusion':
            l.trainable = True
        elif l.name == 'resnet50':
            l.trainable = False
        print(l.name, l.trainable)


    return model









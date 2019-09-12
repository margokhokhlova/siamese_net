from keras.applications import ResNet50

from keras.layers import GlobalMaxPooling2D, Dense, Conv2D, Lambda
from keras.models import Model,  Sequential, Input
from keras.applications import imagenet_utils
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-imsize", "--img-size", default = 512, help="Input Image dimensions")
ap.add_argument("-c", "--channels", type=int, default=3,
    help="number of channels in the image: RGB = 3, RGBI = 4, etc")
ap.add_argument("-p", "--pooling", type=bool, default=True,
    help="with max pool or no?")
args = vars(ap.parse_args())
H = W = args['img_size']
C = args['channels']
pooling = args["pooling"]



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
        model_top.add(Conv2D(filters=3, kernel_size=(1,1), strides=(1, 1), activation = 'relu', padding = 'same', input_shape=(H,W,C)))
        model_top.add(Lambda(imagenet_utils.preprocess_input)) # custom pre-processing layer
    model_top.add(Dense(3, activation=None, input_shape=(H,W,C)))
    baseModel = ResNet50(weights=weights, include_top=False)
    model_top.add(baseModel)
    model_top.summary()
    # model_top.add(baseModel)
    # add the global max pooling layer
    headModel = model_top.output
    headModel = GlobalMaxPooling2D()(headModel) # add a GLOBAL MAXPOOLING layer to obtain one value per channel
    if pooling:
        model = Model(inputs = model_top.input, outputs=headModel)
        shape_feat =  2048

    else:
        model = model_top
        shape_feat = 16 * 16 * 2048

    model.summary()
    return model









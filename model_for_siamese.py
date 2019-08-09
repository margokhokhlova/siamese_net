from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import GlobalMaxPooling2D, Dense
from keras.models import Model,  Sequential, Input
from imutils import paths
import numpy as np
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
def get_model (H,W,C, pooling =True, weights = 'imagenet'):
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







# dataset things
# bs = args["batch_size"]
#
# # grab all image paths in the input directory and randomly shuffle
# # the paths
# imagePaths = list(paths.list_images(args["dataset"]))
# imagePaths.sort() #sort in alphabetical order
# print(len(imagePaths))
#
# # extract the class labels from the image paths, then encode the
# # labels
# labels = imagePaths
#
#
# # define our set of columns
# cols = ["feat_{}".format(i) for i in range(0, shape_feat)]
# cols = ["path"] +["vector"] + cols
#
# # open the CSV file for writing and write the columns names to the
# # file
# csv = open(args["csv"], "w")
# csv.write("{}\n".format(",".join(cols)))
#
# # loop over the images in batches
# for (b, i) in enumerate(range(0, len(imagePaths), bs)):
#     # extract the batch of images and labels, then initialize the
#     # list of actual images that will be passed through the network
#     # for feature extraction
#     print("[INFO] processing batch {}/{}".format(b + 1,
#         int(np.ceil(len(imagePaths) / float(bs)))))
#     batchPaths = imagePaths[i:i + bs]
#     batchLabels = labels[i:i + bs]
#     batchImages = []
#
#     # loop over the images and labels in the current batch
#     for i in range(0,len(batchPaths)-1,2):
#
#         # load the input image using the Keras helper utility while
#         # ensuring the image is resized to 224x224 pixels
#         #print(imagePath)
#         image = load_img(imagePaths[i], target_size=(512, 512))
#         lbl = load_img(imagePaths[i+1], target_size=(512, 512))
#         image = img_to_array(image)
#         lbl = np.rot90(lbl, k=1, axes=(1, 0)) # this line turns the image 90 clockwise, needed for IGN data
#
#         # preprocess the image according to image net utils data
#         #image = np.expand_dims(image, axis=0)
#         image = imagenet_utils.preprocess_input(image)
#
#         # preprocess the label image by creating an R,G,B, 1...N features image featuring
#         #lbl_channels =  create_masks_from_rgb(lbl)
#
#         #pre-process image by just making it a grayscale image
#         gray = np.expand_dims(np.mean(lbl, -1), axis = 2)# expand
#
#     # concatenate the resulting images horizontally
#         image = np.dstack((image, gray))
#         image = np.expand_dims(image, axis=0)
#         # add the image to the batch
#         batchImages.append(image)
#
#     # pass the images through the network and use the outputs as our
#     # actual features, then reshape the features into a flattened
#     # volume
#     batchImages = np.vstack(batchImages)
#     features = model.predict(batchImages, batch_size=bs)
#     features = features.reshape((features.shape[0], shape_feat)) #2048 - filter size, image 224 -> 7
#
#     # loop over the class labels and extracted features
#     for (label, vec) in zip(batchLabels, features):
#         # construct a row that exists of the class label and extracted
#         # features
#         # extract the image type -> vector or semantic
#         class_lbl = 1 if label[-7:-4] == 'lbl' else 0
#
#
#         vec = ",".join([str(v) for v in vec])
#         csv.write("{},{},{}\n".format(label,class_lbl, vec))
#
# # close the CSV file
# csv.close()
# # python3 -m cirtorch.examples.extract_features --network-offtheshelf 'resnet101-gem-reg-whiten' --datasets oxford5k --image-size 1024 --csv 'descriptor_test.csv'

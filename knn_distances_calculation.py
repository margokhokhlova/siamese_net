from dataloader_fused import fused_datagenerator
import tensorflow as tf
import os
from keras import backend as K
import numpy as np
from dataloader_fused import fused_datagenerator
from sklearn import preprocessing

from sklearn.neighbors import NearestNeighbors
from evaluate import map_for_dataset, get_hard

def test_labels_check(keys_2019, keys_2004):
    ''' returns the confirmed gt labels 2019-2004 (checks the images names) '''
    keys_short_19 = get_keys_as_image_coordinates(keys_2019)
    keys_short_04 = get_keys_as_image_coordinates(keys_2004)
    gt_indexes = []
    for key19 in keys_short_19:
        try:
         gt_indexes.append(keys_short_04.index(key19))

        except ValueError:
            print('cannot find the index %s' % (key19))
            break
    return gt_indexes


def get_keys_as_image_coordinates(keys):
    ''' instead of a full path, return only image coordinates + index'''
    modif_keys = [key[-12:-7] + key[-37:-28] for key in keys]
    # modif_keys = []
    # for key in keys:
    #      modif_keys.append(key[-12:-7] + key[-37:-28])
    return modif_keys


def scale_features(feat_vector):
    feat_vector = np.asarray(feat_vector)
    feat_vector = feat_vector.reshape(-1,2048)
    X_scaled = preprocessing.normalize(feat_vector)
    return X_scaled

def knn_distance_calculation(model, path_2004, path_2019, bs =10):
    topo_ortho_generator2019 = fused_datagenerator(dataset_im = path_2019, batch_size = bs, n_channel_lbl=1)
    topo_ortho_generator2004 = fused_datagenerator(dataset_im = path_2004, batch_size = bs, n_channel_lbl=1)

    topo_ortho_generator2019.getImagePaths()
    topo_ortho_generator2004.getImagePaths()

    features2019 = []
    features2004 = []
    labels2019 = []
    labels2004 = []
    # loop over the images in batches
    actual_batch_size = int(bs / 2)
    for (b, i) in enumerate(range(int(topo_ortho_generator2019.total_images/bs))):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        print("[INFO] processing batch {}/{}".format(b + 1,
            int(np.ceil(topo_ortho_generator2019.total_images / float(bs)))))


        # pass the images through the network and use the outputs as our
        # actual features, then reshape the features into a flattened
        # volume

        X, batchLabels = topo_ortho_generator2019.__getitem__(i)
        features = model.predict(X, batch_size= actual_batch_size) # 2019
        features2019.append(features.reshape(features.shape[0],2048))
        labels2019 = np.hstack([labels2019, batchLabels])

        X, batchLabels = topo_ortho_generator2004.__getitem__(i)
        features = model.predict(X, batch_size=actual_batch_size)  # 2019
        features2004.append(features.reshape(features.shape[0], 2048))
        labels2004 = np.hstack([labels2004, batchLabels])
    # here I should already have all the features extracted

    # gt_indexes = test_labels_check(features2019, features2004) #check order
    data_base = scale_features(features2004)
    query = scale_features(features2019)

    gt_indexes = np.arange(int(topo_ortho_generator2019.total_images/2))
    # knn
    knn_array = []
    dist_array = []
    # fit the K-nn algo
    neigh = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neigh.fit(data_base)
    num_img = len(data_base)
    for i in range(num_img):
        dist, indexes = neigh.kneighbors([query[i, :]])
        knn_array.append(indexes)  # workaround for structure
        dist_array.append(dist)
    print("total of %d images were processed" % i)
    map = map_for_dataset(gt_indexes, knn_array, dist_array)
    print("Final MAP for the data is %f" % (map))

    hard_pairs = get_hard(gt_indexes, knn_array,  labels2004)
    return hard_pairs
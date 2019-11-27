import numpy as np
from dataloader_fused import fused_datagenerator


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


def knn_distance_calculation(model, path_2004, path_2019, bs =10, feat_shape = 512, im_size = (512,512)):
    """ make a dummy run with a base model on the whole dataset and calculate KNN map@5"""
    topo_ortho_generator2019 = fused_datagenerator(dataset_im = path_2019, n_channel_lbl=1,  batch_size = bs,target_size =  im_size)
    topo_ortho_generator2004 = fused_datagenerator(dataset_im = path_2004, n_channel_lbl=1,  batch_size = bs, target_size =  im_size)

    topo_ortho_generator2019.getImagePaths()
    topo_ortho_generator2004.getImagePaths()


    features2019 = []
    features2004 = []
    labels2019 = []
    labels2004 = []
    # loop over the images in batches

    for (b, i) in enumerate(range(int(topo_ortho_generator2019.total_images/bs))):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        #print("[INFO] processing batch {}/{}".format(b + 1,
        #    int(np.ceil(topo_ortho_generator2019.total_images / float(bs)))))


        # pass the images through the network and use the outputs as our
        # actual features, then reshape the features into a flattened
        # volume

        X, batchLabels = topo_ortho_generator2019.__getitem__(i)
        features = model.predict(X, batch_size= bs) # 2019
        features2019.append(features.reshape(features.shape[0], feat_shape))
        labels2019 = np.hstack([labels2019, batchLabels])

        X, batchLabels = topo_ortho_generator2004.__getitem__(i)
        features = model.predict(X, batch_size=bs)  # 2019
        features2004.append(features.reshape(features.shape[0], feat_shape))
        labels2004 = np.hstack([labels2004, batchLabels])
    # here I should already have all the features extracted

    # gt_indexes = test_labels_check(features2019, features2004) #check order
    data_base = np.asarray(features2004).reshape(-1,feat_shape)
    query = np.asarray(features2019).reshape(-1,feat_shape)
    assert(data_base.shape[0] == query.shape[0]), 'The database and query size is not equal in Knn dist calc file'
    num_img = len(data_base)
    for i in range(num_img):
        assert(labels2019[i][-31:]==labels2004[i][-31:]), 'Mismatch in the database labels, check it!'
    gt_indexes = np.arange(query.shape[0])
    # knn
    knn_array = []
    dist_array = []
    # fit the K-nn algo
    neigh = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    neigh.fit(data_base)

    for i in range(num_img):
        dist, indexes = neigh.kneighbors([query[i, :]])
        knn_array.append(indexes)  # workaround for structure
        dist_array.append(dist)
    print("total of %d images were processed" % i)
    map = map_for_dataset(gt_indexes, knn_array, dist_array)
    print("Final MAP for the data is %f" % (map))

    hard_pairs = get_hard(gt_indexes, knn_array,  labels2004, labels2019)
    return hard_pairs, map
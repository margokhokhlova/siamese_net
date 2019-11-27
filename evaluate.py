import numpy as np


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]

        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map_per_class(ranks, gnd, classes=None, kappas=[]):
    """
    Computes the mAP per class.
    If we don't know the class per image, it can be guessed
    """

    maps = {}
    aps = {}

    if 'classes' is not None:
        unique_classes = np.unique(classes)
    else:
        #TODO
        raise NotImplementedError

    start = 0
    for currentclass in unique_classes:
        nb_queries = np.sum(classes == currentclass)
        classranks = ranks[:, start:start+nb_queries]
        maps[currentclass], aps[currentclass], _, _ = compute_map(classranks, gnd[start:start+nb_queries])
        start += nb_queries

    return maps, aps


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10], writer=None, epoch=0):
    
    # old evaluation protocol
    if dataset in ['oxford5k', 'paris6k', 'skraa', 'holidays', 'ukbench', 'alegoria']:
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))

        if writer is not None:
            writer.add_scalar("test/mAP_{}".format(dataset), map, epoch)

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

        if writer is not None:
            writer.add_scalar("test/mAP_{}_easy".format(dataset), mapE, epoch)
            writer.add_scalar("test/mAP_{}_medium".format(dataset), mapM, epoch)
            writer.add_scalar("test/mAP_{}_hard".format(dataset), mapH, epoch)





def map_for_dataset(gt, returned_queries, distances = None):
    ''' My implementation for the MAP, when the correct image is always a single image
    input - list of gt labels gt and returned queries - list of lists of returned queries '''
    assert len(gt)==len(returned_queries), "The number of HT indexes is not equal to the number of returned values!"
    av_pr = 0
    images_to_ignore = 0
    for i in range(len(gt)):
        query = np.ravel(returned_queries[i]).tolist()
        gt_query = gt[i]
        try:
            element_rank =query.index(gt_query) + 1  #modify the query according to the gt
        except:
            element_rank = 0


        # if distances is not None:
            # don't count the sample if all the distances are same - probably, it is a white image
            # if abs(distances[i][0][0] - distances[i][0][1] + (distances[i][0][1]-distances[i][0][2])) < 0.000000002:
            #     images_to_ignore +=1
            #     continue
        av_pr += 1/element_rank if element_rank > 0 else 0


    # print("Ignored %d empty images" %images_to_ignore)
    return av_pr/(len(gt))



def mathing_index_for_one_image(gt_idx, returned_queries_idx, distances = None):
    ''' This function just returns the matching coefficient if the returned query image matches '''
    correct_match = list(returned_queries_idx).index(gt_idx)
    return correct_match

def get_hard(gt_idx, returned_query_indexes, image_labels2004, image_labels2019):
    """ function creates the dictionaty with image indexes and corresponding
    'hard' pairs -> images which were neares neightbors in a wrongly located image
    input: gt_index - list of the gt geolocalized images
            returned_queries - knn returned
            images_labels paths to images
    output:
    dictionary -> images_label  as key and correponding hard neightbors as values"""

    hard_pairs = {}
    for i in range(0,len(gt_idx)):
        if returned_query_indexes[i][0][0] != gt_idx[i] and returned_query_indexes[i][0][1] != gt_idx[i]: # if the correct match is not the first 2 elements in the query: can be replaced with el in list
            hard_pairs[image_labels2019[i], image_labels2004[i]] = [image_labels2004[j] for j in returned_query_indexes[i][0] if j!=i]
    return hard_pairs

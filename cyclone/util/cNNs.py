import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical
import random
from annoy import AnnoyIndex
import scipy.sparse as sps
from itertools import product

SCALE_BEFORE = True
EPS = 1e-12
VERBOSE = False

def svd1(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def pls(x, y, num_cc):
    random.seed(42)
    plsca = PLSCanonical(n_components=int(num_cc), algorithm='svd')
    fit = plsca.fit(x, y)
    u = fit.x_weights_
    v = fit.y_weights_
    a1 = np.matmul(np.matrix(x), np.matrix(u)).transpose()
    d = np.matmul(np.matmul(a1, np.matrix(y)), np.matrix(v))
    ds = [d[i, i] for i in range(0, 30)]
    return u, v, ds

def scale_nor(x):
    # y = preprocessing.scale(x)  # scale each col separately
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y = (x - x_mean) / (x_std + EPS)

    return y

def runcca(data1, data2, num_cc=20, scale_before=SCALE_BEFORE):
    random.seed(42)

    object1 = scale_nor(data1) if scale_before else data1.copy()
    object2 = scale_nor(data2) if scale_before else data2.copy()

    mat3 = object1.T.dot(object2)
    a = svd1(mat=mat3, num_cc=num_cc)
    cca_data = np.concatenate((a[0], a[1]))
    ind = np.where(
        [cca_data[:, col][0] < 0 for col in range(cca_data.shape[1])])[0]
    cca_data[:, ind] = cca_data[:, ind] * (-1)

    d = a[2]
    return cca_data, d


def l2norm(mat):
    stats = np.sqrt(np.sum(mat**2, axis=1, keepdims=True)) + EPS
    mat = mat / stats
    return mat


def runCCA(data1, data2, features, num_cc):
    features_idx = np.arange(len(features))
    if VERBOSE:
        print(f'====>{len(features_idx)} left ')

    cca_results = runcca(data1=data1, data2=data2, num_cc=num_cc)
    cell_embeddings = cca_results[0]

    combined_data = np.hstack([data1, data2])
    loadings = combined_data.dot(cell_embeddings)
    return cca_results, loadings, features_idx

#data Input data
#query Data to query against data
#k Number of nearest neighbors to compute
#Approximate nearest neighbors using locality sensitive hashing.
def NN(data, query=None, k=10, metric='manhattan', n_trees=10):
    if query is None:
        query = data

    # Build index.
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    ind = np.array(ind)

    return ind

def findNN(data1, data2, k):
    if VERBOSE:
        print("Finding nearest neighborhoods")
    nnab = NN(data=data2, query=data1, k=k)
    nnba = NN(data=data1, query=data2, k=k)
    return nnab, nnba


def findMNN(neighbors, num):
    max_nn = np.array([neighbors[0].shape[1], neighbors[1].shape[1]])
    if ((num > max_nn).any()):
        num = np.min(max_nn)
        # convert cell name to neighbor index
    
    nnab, nnba = neighbors[0], neighbors[1]

    pairs_ab, pairs_ba = set(), set()
    # build set of mnn of (b1, b2)
    for i,nni in enumerate(nnab):
        nni = nni[:num]  # take the top num neighbors
        for j in nni:
            pairs_ab.add((i, j))

    for i, nni in enumerate(nnba):
        nni = nni[:num]
        for j in nni:
            pairs_ba.add((j, i))

    pairs = pairs_ab & pairs_ba
    pairs = np.array([[p[0], p[1]] for p in pairs])
    
    mnns = pd.DataFrame(pairs, columns=['cell1', 'cell2'])
    if VERBOSE:
        print(f'\t Found {mnns.shape[0]} mnn pairs')
    return mnns

# norm_list: [np.array] * N, list of normalized data
# features: array, highly variable features, array
# cname_list: [np.array] * N, list of cell names of each batch
# num_cc:  int,  dim of cca
# k_filter: int, knn for filtering
def generate_graph(norm_list, cname_list, features, combine, num_cc=30, k_filter=200, k_neighbor=5, filtering=False):
    all_pairs = []
    for row in combine:
        i = row[0]
        j = row[1]
        norm_data1 = norm_list[i]
        norm_data2 = norm_list[j]

        cell_embedding, loading, features_filtered = runCCA(data1=norm_data1,    # scale and runcca
                                         data2=norm_data2,
                                         features=features,
                                         num_cc=num_cc)
        norm_embedding = l2norm(mat=cell_embedding[0])
        #identify nearest neighbor
        cells1 = cname_list[i]
        cells2 = cname_list[j]
        neighbor = findNN(
                    data1 = norm_embedding[:len(cells1)],
                    data2 = norm_embedding[len(cells1):],
                    k=30)
        #identify mutual nearest neighbors

        mnn_pairs = findMNN(neighbors=neighbor,
                            num=k_neighbor)
        final_pairs = mnn_pairs

        final_pairs['cell1_name'] = final_pairs.cell1.apply(lambda x: cells1[x])
        final_pairs['cell2_name'] = final_pairs.cell2.apply(lambda x: cells2[x])

        final_pairs['Dataset1'] = [i + 1] * final_pairs.shape[0]
        final_pairs['Dataset2'] = [j + 1] * final_pairs.shape[0]
        all_pairs.append(final_pairs)

    all_pairs = pd.concat(all_pairs)
    return all_pairs

'''
    X: array or csr_matrix, 
    batch_label: array,
    cname: array
    gname: array
    sketch: using geosketching to reduce number of cells
    k_anchor: number of K used to select mnn
'''
def computeAnchors(X, batch_label, cname, gname, k_anchor=5, filtering=True):
    norm_list, cname_list = [], []
    bs = np.unique(batch_label)
    n_batch = bs.size

    X = X.copy()

    for bi in bs:
        bii = np.where(batch_label==bi)[0]
        X_bi = X[bii].A if sps.issparse(X) else X[bii]
        cname_bi = cname[bii]

        norm_list.append(X_bi.T)
        cname_list.append(np.array(cname_bi))

    combine = list(product(np.arange(n_batch), np.arange(n_batch)))
    combine = list(filter(lambda x: x[0]<x[1], combine))
    anchors = generate_graph(norm_list, cname_list, gname, combine,
                                num_cc=20, k_filter=200, k_neighbor=k_anchor, filtering=filtering)

    return anchors


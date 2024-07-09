from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from functools import partial
from collections import defaultdict
from util.NNs import nn_approx, reduce_dimensionality
from util.cNNs import computeAnchors

class pairs_dataset(Dataset):
    '''
        unsupervised contrastive batch correction:
        postive anchors: if MNN pairs else augmented by gausian noise
        negtive anchors: all samples in batch except pos anchors
    '''
    def __init__(
            self, 
            adata,
            mode='train',
            alpha=0.9,
            knn = 10,           # used to compute knn
            augment_set=['int', 'geo', 'exc'],  # 'int': interpolation, 'geo': geometric, 'exc': exchange
            exclude_fn=True,
            verbose=0,
            cname = None,
            gname = None
        ):
        self.mode = mode
        self.verbose = verbose
        self.data = adata
        self.augment_op_names = augment_set

        self.alpha = alpha
        self.knn = knn

        self.X = adata.obsm['vae_emb']
        self.metadata = adata.obs
        self.gname = adata.var_names if gname is None else gname
        self.cname = adata.obs_names if cname is None else cname
        self.n_sample = self.X.shape[0]
        self.n_feature = self.X.shape[1]
        self.name2idx = dict(zip(self.cname, np.arange(self.n_sample)))
        self.n_batch = len(adata.obs['Batch'].unique())
        self.batch_label = adata.obs['Batch'].values
        if 'CellType' in adata.obs.columns:
            self.type_label = adata.obs['CellType'].values
            self.n_type = len(self.type_label.unique())
        else:
            self.type_label = None
            self.n_type = None

        # define set of augment operatio
        self.augment_set = []
        for ai in augment_set:
            if ai=='int':
                self.augment_set.append(partial(interpolation, alpha=alpha))
            elif ai=='geo':
                self.augment_set.append(partial(geo_interpolation, alpha=alpha))
            elif ai=='exc':
                self.augment_set.append(partial(binary_switch, alpha=alpha))
            else:
                raise ValueError("Unrecognized augment operation")
        if self.verbose:
            print('Defined ops: ', self.augment_op_names)

        if mode=='train':
            #computing anchors
            self.compute_anchors(self.X, self.batch_label, self.cname, self.gname, k_anchor=5, filtering=True)

            self.getMnnDict()
            self.exclude_sampleWithoutMNN(exclude_fn)

            self.computeKNN(knn)

    def __len__(self):
        if self.mode=='train':
            return len(self.valid_cellidx)
        else:
            return self.X.shape[0]

    def update_pos_nn_info(self):
        # create positive sample index
        rand_ind1 = np.random.randint(0, self.knn, size=(self.n_sample))
        self.rand_nn_ind1 = self.nns[np.arange(self.n_sample), rand_ind1]
        rand_ind2 = np.random.randint(0, self.knn, size=(self.n_sample))

        self.lambdas1 = np.random.uniform(self.alpha, 1, size=(self.n_sample, 1))
        self.lambdas2 = np.random.uniform(self.alpha, 1, size=(self.n_sample, 1))

        self.rand_pos_ind = [np.random.choice(self.mnn_dict[i]) if len(self.mnn_dict[i])>0 else i for i in range(self.n_sample)]

        X_arr = self.X
        X_pos = X_arr[self.rand_pos_ind]
        pos_knns_ind = self.nns[self.rand_pos_ind]
        self.pos_nn_ind = pos_knns_ind[np.arange(self.n_sample), rand_ind2]
        self.X1 = X_arr*self.lambdas1 + X_arr[self.rand_nn_ind1]*(1-self.lambdas1)
        self.X2 = X_pos*self.lambdas2 + X_arr[self.pos_nn_ind] * (1-self.lambdas2)

    def compute_anchors(self, X, batch_label, cname, gname, k_anchor=5, filtering=True):
        print('computeing anchors')
        anchors = computeAnchors(X, batch_label, cname, gname, k_anchor=k_anchor, filtering=filtering)

        anchors.cell1 = anchors.cell1_name.apply(lambda x: self.name2idx[x])
        anchors.cell2 = anchors.cell2_name.apply(lambda x: self.name2idx[x])
        pairs = np.array(anchors[['cell1', 'cell2']])
        self.pairs = pairs

        # print anchor info
        if self.verbose and (self.type_label is not None):
            print_AnchorInfo(self.pairs, self.batch_label, self.type_label)


    def computeKNN(self, knn=10):
        # calculate knn within each batch
        self.nns = np.ones((self.n_sample, knn), dtype='long')  # allocate (N, k+1) space
        bs = self.batch_label.unique()
        for bi in bs:
            bii = np.where(self.batch_label==bi)[0]

            # dim reduction for efficiency
            X_pca = reduce_dimensionality(self.X, 50)
            nns = nn_approx(X_pca[bii], X_pca[bii], knn=knn+1)  # itself and its nns
            nns = nns[:, 1:]

            # convert local batch index to global cell index
            self.nns[bii, :] = bii[nns.ravel()].reshape(nns.shape)

        if self.verbose and (self.type_label is not None):
            print_KnnInfo(self.nns, np.array(self.type_label))

    def exclude_sampleWithoutMNN(self, exclude_fn):
        self.valid_cellidx = np.unique(self.pairs.ravel()) if exclude_fn else np.arange(self.n_sample)

        if self.verbose:
            print(f'Number of training samples = {len(self.valid_cellidx)}')

    def getMnnDict(self):
        self.mnn_dict = get_mnn_graph(self.n_sample, self.pairs)


    def getTrainItem(self, i):
        i = self.valid_cellidx[i] if self.mode=='train' else i
        pi = self.rand_pos_ind[i]
        
        x_aug = self.X1[i]
        x_p_aug = self.X2[i]

        return [x_aug.astype('float32'), x_p_aug.astype('float32')], [i, pi]

    def getValItem(self, i):
        x = self.X[i].A.squeeze()

        return [x.astype('float32'), x.astype('float32')], [i, i]

    def __getitem__(self, i):
        if self.mode=='train':
            return self.getTrainItem(i)
        else:
            return self.getValItem(i)


# utils
def get_mnn_graph(n_cells, anchors):
    mnn_dict = defaultdict(list)
    for r,c in anchors:   
        mnn_dict[r].append(c)
        mnn_dict[c].append(r)
    return mnn_dict

def interpolation(x, x_p, alpha):
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = lamda * x + (1 - lamda) * x_p
    return x

def geo_interpolation(x, x_p, alpha):
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = (x**lamda) * (x_p**(1-lamda))
    return x

def binary_switch(x, x_p, alpha):
    bernou_p = bernoulli.rvs(alpha, size=len(x))
    x = x * bernou_p + x_p * (1-bernou_p)
    return x

def augment_positive(ops, x, x_p):
    # op_i = np.random.randint(0, 3)
    if len(ops)==0:  # if ops is empty, return x
        return x

    opi = np.random.randint(0, len(ops))
    sel_op = ops[opi]

    return sel_op(x, x_p) 

def print_AnchorInfo(anchors, global_batch_label, global_type_label):
    anchor2type = np.array(global_type_label)[anchors]
    correctRatio = (anchor2type[:, 0] == anchor2type[:, 1]).sum() / len(anchors)
    print('Anchors n={}, ratio={:.4f}'.format(len(anchors), correctRatio))

    anchors = anchors.ravel()
    df = pd.DataFrame.from_dict({"type": list(global_type_label[anchors]), 'cidx':anchors,
                                "batch": list(global_batch_label[anchors])},
                                orient='columns')
    print(df.groupby('batch')['cidx'].nunique() / global_batch_label.value_counts())
    print(df.groupby('type')['cidx'].nunique() / global_type_label.value_counts())


def print_KnnInfo(nns, type_label, verbose=0):
    def sampleWise_knnRatio(ti, nn, tl):
        knn_ratio = ti == tl[nn]
        knn_ratio = np.mean(knn_ratio)
        return knn_ratio

    if isinstance(nns, defaultdict):
        corr_ratio_per_sample = []
        for k,v in nns.items():
            corr_ratio_per_sample.append(np.mean(type_label[k] == type_label[v]))
    else:
        corr_ratio_per_sample = list(map(partial(sampleWise_knnRatio, tl=type_label), type_label, nns))

    ratio = np.mean(corr_ratio_per_sample)
    print('Sample-wise knn ratio={:.4f}'.format(ratio))

    if verbose:
        return corr_ratio_per_sample
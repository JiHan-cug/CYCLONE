import random
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import linear_sum_assignment

def make_dataloader(adata, con_data, mnn_index, knn_index_1, knn_index_2, batch_size, alpha=0.8):
    feature = torch.tensor(adata.X)
    label = con_data.batch_label
    mnn = feature[mnn_index]

    n_sample = feature.shape[0]
    lambdas1 = np.random.uniform(alpha, 1, size=(n_sample, 1))
    lambdas2 = np.random.uniform(alpha, 1, size=(n_sample, 1))

    X1 = torch.tensor(np.array(feature) * lambdas1 + np.array(feature)[knn_index_1] * (1 - lambdas1))
    X2 = torch.tensor(np.array(mnn) * lambdas2 + np.array(feature)[knn_index_2] * (1 - lambdas2))

    OneHot_enc = OneHotEncoder()
    onehot_code = OneHot_enc.fit_transform(np.array(label).reshape(-1, 1))
    batch_onehot = torch.tensor(onehot_code.toarray())

    dataset = TensorDataset(feature, torch.tensor(label), batch_onehot, X1, X2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def calculate_metric(pred, label):
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return nmi, ari

def calculate_BASW(adata, emb_key='feat', batch_key='Batch'):
    np.random.seed(2023)
    asw_batch = silhouette_score(adata.obsm[emb_key], adata.obs[batch_key])
    asw_batch_norm = abs(asw_batch)

    BASW = 1 - asw_batch_norm
    return BASW

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_ber(label, pred, batch):
    labels_name = np.unique(label)
    label_dict = {name:i for i,name in enumerate(labels_name)}
    label = np.array([label_dict[x] for x in label])
    y_true = np.array(label)
    y_pred = np.array(pred)
    y_true = y_true.astype(np.float64).astype(np.int64)
    y_pred = y_pred.astype(np.float64).astype(np.int64)

    assert y_pred.size == y_true.size
    K = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((K, K), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind,col_ind = linear_sum_assignment(w.max() - w)

    for i in range(len(label)):
        y_pred[i] = col_ind[y_pred[i]]

    batch_name = np.unique(batch)
    B = len(batch_name)
    batch_dict = {name: i for i, name in enumerate(batch_name)}
    batch = np.array([batch_dict[x] for x in batch])
    batch = batch.astype(np.float64).astype(np.int64)

    pred_mx = np.zeros((K, B))
    true_mx = np.zeros((K, B))
    for k in range(K):
        for b in range(B):
            pred_mx[k, b] = np.sum((y_pred == k) & (batch == b))
            true_mx[k, b] = np.sum((y_true == k) & (batch == b))

    ber = np.sum(pred_mx * np.log((pred_mx+1) / (true_mx+1)))

    return ber



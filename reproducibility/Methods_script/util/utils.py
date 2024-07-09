import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
import pandas as pd
import scipy.sparse as sps
import pickle
from os.path import join
import warnings
warnings.filterwarnings("ignore")

label_encoder = LabelEncoder()

def data_preprocess(adata, batch_key, label_key, select_hvg=None, scale=False):
    adata.obs['Batch'] = adata.obs[batch_key]
    adata.obs['celltype'] = adata.obs[label_key]

    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if select_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(adata.shape[1], select_hvg),
                                    batch_key='Batch')

        adata = adata[:, adata.var.highly_variable].copy()

    if scale:
        warnings.warn('Scaling per batch! This may cause memory overflow!')
        ada_batches = []
        for bi in adata.obs['Batch'].unique():
            bidx = adata.obs['Batch'] == bi
            adata_batch = adata[bidx].copy()
            sc.pp.scale(adata_batch)

            ada_batches.append(adata_batch)

        adata = sc.concat(ada_batches)


    return adata

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

def py_read_data(_dir, fname):

    sps_X = sps.load_npz(join(_dir, fname+'.npz'))    # read gene names
    with open(join(_dir, fname+'_genes.pkl'), 'rb') as f:
        genes = pickle.load(f)

    # read cell names
    with open(join(_dir, fname+'_cells.pkl'), 'rb') as f:
        cells = pickle.load(f)

    return sps_X, cells, genes


def load_meta_txt(path, delimiter='\t'):
    data, colname, cname = [], [], []
    with open(path, 'r') as f:
        for li, line in enumerate(f):
            line = line.strip().replace("\"", '').split(delimiter)

            if li == 0:
                colname = line
                continue

            cname.append(line[0])

            data.append(line[1:])
    df = pd.DataFrame(data, columns=colname, index=cname)
    return df

#for Human PBMC adata
def adata_pbmc(data_root):
    sps_x1, gene_name1, cell_name1 = py_read_data(data_root, 'b1_exprs')
    sps_x2, gene_name2, cell_name2 = py_read_data(data_root, 'b2_exprs')

    sps_x = sps.hstack([sps_x1, sps_x2])

    df_meta1 = load_meta_txt(join(data_root, 'b1_celltype.txt'))
    df_meta2 = load_meta_txt(join(data_root, 'b2_celltype.txt'))
    df_meta1['batchlb'] = 'Batch1'
    df_meta2['batchlb'] = 'Batch2'

    df_meta = pd.concat([df_meta1, df_meta2])

    df_meta['batchlb'] = df_meta['batchlb'].astype('category')
    df_meta['CellType'] = df_meta['CellType'].astype('category')

    dense_x = sps_x.toarray().T
    adata = sc.AnnData(X=dense_x)
    adata.obs['batch'] = np.array(df_meta['batchlb'])
    adata.obs['celltype'] = np.array(df_meta['CellType'])
    return adata

#for MCA adata
def adata_mca(data_root):
    sps_x, gene_name, cell_name = py_read_data(data_root, 'filtered_total_batch1_seqwell_batch2_10x')
    df_meta = load_meta_txt(join(data_root, 'filtered_total_sample_ext_organ_celltype_batch.txt'))
    df_meta['CellType'] = df_meta['ct']

    df_meta['batchlb'] = df_meta['batchlb'].astype('category')
    df_meta['CellType'] = df_meta['CellType'].astype('category')

    df_meta['CellType'] = df_meta['ct']

    df_meta['batchlb'] = df_meta['batchlb'].astype('category')
    df_meta['CellType'] = df_meta['CellType'].astype('category')
    dense_x = sps_x.toarray().T
    adata = sc.AnnData(X=dense_x)
    adata.obs['batch'] = np.array(df_meta['batchlb'])
    adata.obs['celltype'] = np.array(df_meta['CellType'])
    return adata

#for seurat input
pbmc = adata_pbmc('../data/Human PBMC')
pbmc.write_h5ad('../data/PBMC.h5ad')

#for seurat input
mca = adata_mca('../data/MCA')
mca.write_h5ad('../data/MCA.h5ad')
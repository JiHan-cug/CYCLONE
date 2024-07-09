import scanpy as sc
import pandas as pd
import anndata as ad
import time
import numpy as np
from time import time
from memory_profiler import profile
from sklearn.preprocessing import LabelEncoder

@profile
def my_func(adata):
    start = time()
    cell_type = adata.obs['celltype'].values
    Label_enc = LabelEncoder()
    cell_type = Label_enc.fit_transform(cell_type)
    batch = adata.obs['tech'].values
    Batch = LabelEncoder()
    batch = Batch.fit_transform(batch)

    adata.obs['celltype'] = cell_type
    adata.obs['celltype'] = adata.obs['celltype'].astype('int').astype('category')
    adata.obs['batch'] = batch
    adata.obs['batch'] = adata.obs['batch'].astype('int').astype('category')

    sc.tl.pca(adata, svd_solver="arpack")
    adata.obs['batch'] = adata.obs['batch'].astype('int').astype('category')
    sc.external.pp.bbknn(adata, batch_key='batch', use_rep='X_pca')
    adata.obsm['X_latent'] = adata.obsm['X_pca']
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))

    return adata

if __name__ == '__main__':
    data = sc.read_h5ad('../../data/Human Pancreas/human_pancreas_norm_complexBatch.h5ad')
    data.obs_names_make_unique()
    adata = my_func(data)
    corrd = pd.DataFrame(adata.obsm['X_latent'])
    adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
    adata_corrd.obsm['X_latent'] = adata.obsm['X_latent']
    adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_corrd.obs['batch'] = np.array(adata.obs['tech'])

    adata_corrd.write_h5ad('pancreas_bbknn_emb.h5ad')





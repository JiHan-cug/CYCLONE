from tnn import TNN
from memory_profiler import profile
from time import time
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@profile
def my_func(adata):
    start = time()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat')
    highly_variable = adata.var['highly_variable']
    adata = adata[:, highly_variable]
    sc.tl.pca(adata, n_comps=50)

    model = TNN(k=10, embedding_dims=2, batch_size=64, n_epochs_without_progress=10, verbose=0,
                epochs=200, k_to_m_ratio=0.75, approx=False)
    model.fit(X=adata, batch_name="BATCH", shuffle_mode=True)
    adata.obsm['X_latent'] = model.transform(adata)
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return adata

if __name__ == '__main__':
    data = sc.read_h5ad('../../data/Bct-s/Bct-s.h5ad')
    data.obs_names_make_unique()
    adata = my_func(data)
    corrd = pd.DataFrame(adata.obsm['X_latent'])
    adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
    adata_corrd.obsm['X_latent'] = adata.obsm['X_latent']
    adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_corrd.obs['batch'] = np.array(adata.obs['BATCH'])

    adata_corrd.write_h5ad('Bct-s_insct_emb.h5ad')

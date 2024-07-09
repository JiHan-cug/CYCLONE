import scanpy as sc
import harmonypy as hm
import pandas as pd
import numpy as np
from time import time
import anndata as ad
import sys
sys.path.append("..")
from util.utils import *
from memory_profiler import profile

@profile
def my_func(adata):
    start = time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=20)
    ho = hm.run_harmony(np.array(adata.obsm['X_pca']), adata.obs, ['batch'])
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return ho

if __name__ == '__main__':
    adata = adata_pbmc('../../data/Human PBMC')
    adata.obs_names_make_unique()
    ho = my_func(adata)
    corrd = pd.DataFrame(ho.Z_corr.T)
    adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
    adata_corrd.obsm['X_latent'] = adata_corrd.X
    adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_corrd.obs['batch'] = np.array(adata.obs['batch'])

    adata_corrd.write_h5ad("pbmc_harmony_emb.h5ad")

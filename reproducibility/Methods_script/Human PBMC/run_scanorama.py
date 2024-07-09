import scanorama
import scanpy as sc
import numpy as np
import time
import pandas as pd
import anndata as ad
from memory_profiler import profile
from time import time
import sys
sys.path.append("..")
from util.utils import *
@profile
def my_func(adata):
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=600)
    cell_norms = np.linalg.norm(adata.X, axis=1, ord=2)
    adata.X /= cell_norms[:, None]

    adata_corr = []
    genes_ = []
    all_batch = list(set(adata.obs['batch']))
    for b in all_batch:
        adata_corr.append(adata.X[adata.obs['batch'] == b, :])
        genes_.append(adata.var_names)

    start = time()
    integrated, corrected, genes = scanorama.correct(adata_corr, genes_, return_dimred=True)
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return integrated

if __name__ == '__main__':
    adata = adata_pbmc('../../data/Human PBMC')
    integrated = my_func(adata)
    scanorama_res = np.concatenate(integrated)
    inted = pd.DataFrame(scanorama_res)
    adata_inted = ad.AnnData(inted, obs=adata.obs, dtype='float64')
    adata_inted.obsm['X_latent'] = adata_inted.X
    adata_inted.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_inted.obs['batch'] = np.array(adata.obs['batch'])

    adata_inted.write_h5ad("pbmc_scanorama_emb.h5ad")



import scanorama
import scanpy as sc
import numpy as np
import time
import pandas as pd
import anndata as ad
from memory_profiler import profile
from time import time
@profile
def my_func(adata):
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_genes=600)
    cell_norms = np.linalg.norm(adata.X, axis=1, ord=2)
    adata.X /= cell_norms[:, None]

    adata_corr = []
    genes_ = []
    all_batch = list(set(adata.obs['BATCH']))
    for b in all_batch:
        adata_corr.append(adata.X[adata.obs['BATCH'] == b, :])
        genes_.append(adata.var_names)

    start = time()
    integrated, corrected, genes = scanorama.correct(adata_corr, genes_, return_dimred=True)
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return integrated

if __name__ == '__main__':
    adata = sc.read_h5ad('../../data/Bct-s/Bct-s.h5ad')

    integrated = my_func(adata)

    scanorama_res = np.concatenate(integrated)
    inted = pd.DataFrame(scanorama_res)
    adata_inted = ad.AnnData(inted, obs=adata.obs, dtype='float64')
    adata_inted.obsm['X_latent'] = adata_inted.X
    adata_inted.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_inted.obs['batch'] = np.array(adata.obs['BATCH'])

    adata_inted.write_h5ad("Bct-s_scanorama_emb.h5ad")



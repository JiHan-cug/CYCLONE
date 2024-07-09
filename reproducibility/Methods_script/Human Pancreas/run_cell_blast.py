import scanpy as sc
import os
import numpy as np
import Cell_BLAST as cb
from memory_profiler import profile
from time import time
import anndata as ad
import pandas as pd

os.environ['PYTHONHASHSEED'] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
cb.config.N_JOBS = 4
cb.config.RANDOM_SEED = 0

@profile
def my_func(adata):
    start = time()
    adata.var["variable_genes"].sum()
    model = cb.directi.fit_DIRECTi(adata, genes=adata.var.query("variable_genes").index.to_numpy(),
                                   batch_effect="tech", latent_dim=10, cat_dim=20)
    adata.obsm['X_latent'] = model.inference(adata)
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return adata

if __name__ == '__main__':
    data = sc.read_h5ad('../../data/Human Pancreas/human_pancreas_norm_complexBatch.h5ad')
    adata = my_func(data)
    corrd = pd.DataFrame(adata.obsm['X_latent'])
    adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
    adata_corrd.obsm['X_latent'] = adata.obsm['X_latent']
    adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_corrd.obs['batch'] = np.array(adata.obs['tech'])

    adata_corrd.write_h5ad('pancreas_cell_blast_emb.h5ad')


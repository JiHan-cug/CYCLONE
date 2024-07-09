import scDML
print(scDML.__version__)
import scanpy as sc
from scDML import scDMLModel
import os
from memory_profiler import profile
import numpy as np
from time import time
import anndata as ad
import pandas as pd
import sys
sys.path.append("..")
from util.utils import *
os.environ['PYTHONHASHSEED'] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@profile
def my_func(adata):
    start = time()
    ncluster = 3
    adata.raw = adata
    scdml = scDMLModel(save_dir="./result")
    adata = scdml.preprocess(adata, batch_key="batch", cluster_method="louvain", resolution=3.0)
    scdml.integrate(adata, batch_key="batch", ncluster_list=[ncluster],
                    expect_num_cluster=ncluster, merge_rule="rule2")
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return adata

if __name__ == '__main__':
    adata = adata_pbmc('../../data/Human PBMC')
    adata.obs_names_make_unique()
    adata = my_func(adata)
    corrd = pd.DataFrame(adata.obsm['X_emb'])
    adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
    adata_corrd.obsm['X_latent'] = adata.obsm['X_emb']
    adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_corrd.obs['batch'] = np.array(adata.obs['batch'])

    adata_corrd.write_h5ad('pbmc_scdml_emb.h5ad')


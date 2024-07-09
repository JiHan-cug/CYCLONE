import scanpy as sc
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from memory_profiler import profile
import numpy as np
import pandas as pd
import anndata as ad
import scvi
from sklearn.preprocessing import LabelEncoder
from time import time

@profile
def my_func(adata):
    start = time()
    cell_type = adata.obs['celltype'].values
    Label_enc = LabelEncoder()
    cell_type = Label_enc.fit_transform(cell_type)
    adata.obs['celltype'] = cell_type
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')

    batch_type = adata.obs['BATCH'].values
    Label_enc = LabelEncoder()
    batch_type = Label_enc.fit_transform(batch_type)
    adata.obs['batch'] = batch_type
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="BATCH")
    vae = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30, n_hidden=128)
    vae.train()

    adata.obsm["X_scVI"] = vae.get_latent_representation()
    adata.layers['scvi_normalized'] = vae.get_normalized_expression(library_size=1e4)
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    return adata

if __name__ == '__main__':
    data = sc.read_h5ad('../../data/Bct-s/Bct-s.h5ad')
    adata = my_func(data)
    inted = pd.DataFrame(adata.obsm['X_scVI'])
    adata_inted = ad.AnnData(inted, obs=adata.obs, dtype='float64')
    adata_inted.obsm['X_latent'] = adata.obsm['X_scVI']
    adata_inted.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_inted.obs['batch'] = np.array(adata.obs['BATCH'])
    adata_inted.write_h5ad('Bct-s_scvi_emb.h5ad')

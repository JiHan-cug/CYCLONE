import scanpy as sc
from sklearn.preprocessing import LabelEncoder
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
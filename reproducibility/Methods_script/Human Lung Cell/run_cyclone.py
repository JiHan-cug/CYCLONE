import warnings
import numpy as np
import pandas as pd
import anndata as ad
warnings.filterwarnings("ignore")
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
sc.settings.verbosity = 0
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
from cyclone.train import cyclone
from reproducibility.util.utils import data_preprocess


adata = sc.read_h5ad('../../data/Human Lung Cell/Lung_atlas_public.h5ad')
adata.X = adata.layers['counts'].A

adata = data_preprocess(adata, batch_key='batch', label_key='cell_type', select_hvg=2000, scale=True)


adata, nmi, ari, k, run_time, BASW, BER, mnn_index = cyclone(adata,
                                                                      batch_size=256,
                                                                      vae_z_dim=32,
                                                                      h_dim=16,
                                                                      encode_layers=[1024, 512, 256],
                                                                      decode_layers=[256, 512, 1024],
                                                                      lr_vae=0.0002,
                                                                      train_epoch=30)
types = adata.obs['celltype']
Label_enc = LabelEncoder()
cell_type = Label_enc.fit_transform(types)
z = adata.obsm['cyclone_emb']
y_pred = adata.obs['cyclone_cluster']
corrd = pd.DataFrame(z)
adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
adata_corrd.obsm['X_latent'] = adata.obsm['cyclone_emb']
adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
adata_corrd.obs['batch'] = np.array(adata.obs['Batch'])
adata_corrd.write_h5ad('Bct-s_cyclone_emb.h5ad')

print(f'ARI: {ari}')
print(f'NMI: {nmi}')
print(f'K: {k}')
print(f'BASW: {BASW}')
print(f'BER: {BER}')
print(f'run_time: {run_time}')

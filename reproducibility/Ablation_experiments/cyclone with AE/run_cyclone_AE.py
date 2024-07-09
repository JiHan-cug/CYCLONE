import warnings
import anndata as ad
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import scanpy as sc

sc.settings.verbosity = 0             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

from train import cyclone_AE
from util.preprocess import data_preprocess


adata = sc.read_h5ad('../../data/Simulation.h5ad')

adata = data_preprocess(adata, batch_key='Batch', label_key='Group', select_hvg=2000, scale=True)


adata, nmi, ari, k, run_time, BASW, BER, mnn_index = cyclone_AE(adata,
                                                                      batch_size=256,
                                                                      ae_z_dim=32,
                                                                      h_dim=16,
                                                                      encode_layers=[1024, 512, 256],
                                                                      decode_layers=[256, 512, 1024],
                                                                      lr_ae=0.0002,
                                                                      train_epoch=30)

types = adata.obs['celltype']
Label_enc = LabelEncoder()
cell_type = Label_enc.fit_transform(types)

z = adata.obsm['cyclone_AE_emb']
y_pred = adata.obs['cyclone_AE_cluster']

corrd = pd.DataFrame(z)
adata_corrd = ad.AnnData(corrd, obs=adata.obs, dtype='float64')
adata_corrd.obsm['X_latent'] = adata.obsm['cyclone_AE_emb']
adata_corrd.obs['celltype'] = np.array(adata.obs['celltype'])
adata_corrd.obs['batch'] = np.array(adata.obs['Batch'])
adata_corrd.write_h5ad('Simulation_cyclone_AE_emb.h5ad')

print(f'ARI: {ari}')
print(f'NMI: {nmi}')
print(f'K: {k}')
print(f'BASW: {BASW}')
print(f'BER: {BER}')
print(f'run_time: {run_time}')

###################### This is to calculate the metrics for embedding obtained by the 9 methods on the 6 datasets#################
from utils import *
import anndata as ad
# 6 datasets
data_emb_name = ['Bct-s', 'lung', 'pancreas', 'pbmc', 'mca', 'simulation']
# 9 methods
methods_name = ['bbknn', 'cellblast', 'harmony', 'insct', 'scanorama', 'scdml', 'scvi', 'cyclone',  'seurat']
for method in methods_name:
    if method in ['bbknn', 'cellblast', 'harmony', 'insct', 'scanorama', 'scdml', 'scvi', 'cyclone']:
        for data_name in data_emb_name:
            adata = sc.read_h5ad(f"{data_name}_{method}_emb.h5ad")
            sc.pp.neighbors(adata)
            # Adjust the resolution to achieve the true number of clusters
            sc.tl.leiden(adata, resolution=0.1, random_state=0)
            y_pred = np.asarray(adata.obs['leiden'], dtype=int)
            n_clusters = len(np.unique(y_pred))
            print(f'Number of clusters identified by Louvain for {data_name}_{method} is {n_clusters}')
            nmi, ari = calculate_metric(adata.obs["reassign_cluster"], adata.obs['celltype'])
            adata.obsm['feat'] = adata.obsm['X_latent']
            BASW = calculate_ber(adata)
            BER = calculate_ber(adata.obs['celltype'], y_pred, adata.obs['batch'])
            print(f'NMI= {nmi:.4f}, ARI= {ari:.4f}, BASW= {BASW:.4f}, BER= {BER:.4f}')

    elif method == 'seurat':
        for data_name in data_emb_name:
            csv_data = pd.read_csv(f"{data_name}_seurat_emb.csv")
            assigned_cluster = csv_data['simulate.combined.meta.data...celltype...'].values
            batch = csv_data['simulate.combined.meta.data...batch...'].values
            gene_expression = csv_data.iloc[:, 1:51].values
            adata = ad.AnnData(X=gene_expression, obs={'celltype': assigned_cluster, 'batch': batch})
            cell_type = adata.obs['celltype'].values
            clus_file = pd.read_csv(f"{data_name}_seurat_clust.csv")
            clus = clus_file.iloc[:, 1:].values.flatten().astype(str)
            nmi, ari = calculate_metric(clus, cell_type)
            BASW = calculate_ber(adata)
            BER = calculate_ber(adata.obs['celltype'], clus, adata.obs['batch'])
            print(f'NMI= {nmi:.4f}, ARI= {ari:.4f}, BASW= {BASW:.4f}, BER= {BER:.4f}')

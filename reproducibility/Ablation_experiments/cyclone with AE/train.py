from memory_profiler import profile
from util.tools import *
from model.model import Con_l
from util.loss import *
import time
from sklearn.preprocessing import LabelEncoder
import matplotlib
from util.find_pairs import pairs_dataset
import scanpy as sc
matplotlib.use('Agg')

# @profile
def cyclone_AE(adata,
          batch_size=256,
          ae_z_dim=32,
          h_dim=16,
          encode_layers=[1024, 512, 256],
          decode_layers=[256, 512, 1024],
          lr_ae=0.0002,
          train_epoch=30):
    """
        Train cyclone_AE.
        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        batch_size
            Number of cells for training in one epoch.
        vae_z_dim
            The embedding layer of vae.
        h_dim
            The embedding layer of contrast learning.
        encode_layers
            The hidden layer of encoder
        decode_layers
            The hidden layer of decoder
        lr_vae
            Learning rate for AdamOptimizer.
        train_epoch
            Number of epochs for training.

        Returns
        -------
        adata
            AnnData object of scanpy package. Embedding and clustering result will be stored in adata.obsm['cyclone_AE_emb']
            and adata.obs['cyclone_AE_cluster']
        nmi
            Clustering result NMI.
        ari
            Clustering result ARI.
        K
            The number of clusters by Leiden, If k is not equal to the true number of cell types,
            we can adjust the resolution to the true number of cells within the function `cyclone_AE`
        run_time
            The time it takes for the model training.
        BASW
            BASW value.
        BER
            BER value.
        mnn_index
            A list with the results of each update.
    """
    ####################   Prepare data for training   ####################
    cell_type = adata.obs['celltype'].values
    Label_enc = LabelEncoder()
    cell_type = Label_enc.fit_transform(cell_type)
    batch = adata.obs['Batch'].values
    Batch = LabelEncoder()
    batch = Batch.fit_transform(batch)

    adata.obs['celltype'] = cell_type
    adata.obs['celltype'] = adata.obs['celltype'].astype('int').astype('category')
    adata.obs['Batch'] = batch
    adata.obs['Batch'] = adata.obs['Batch'].astype('int').astype('category')

    y = np.repeat(len(np.unique(batch)), len(batch), axis=0)
    y = np.asarray(y, dtype='int').squeeze()
    adata.obs['domain_number'] = y

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(device)

    OneHot_enc = OneHotEncoder()
    onehot_code = OneHot_enc.fit_transform(np.array(adata.obs['Batch']).reshape(-1, 1))
    batch_onehot = torch.tensor(onehot_code.toarray())

    ############################# Define the model, loss function and optimizer #############################

    AE_model = Con_l(adata.shape[1], adata.obs.domain_number[0], device=device,
                      z_dim=ae_z_dim, h_dim=h_dim, encode_layers=encode_layers, decode_layers=decode_layers).to(device)

    # Contrastive Learning loss
    loss_fn = InstanceLoss(0.07)

    # AE loss
    MSE = nn.MSELoss().to(device)

    optimizer_ae = torch.optim.Adam(AE_model.parameters(), lr=lr_ae)

    ############################# Train of cyclone_AE  #############################

    # hyper-parameter
    con_w = 0.1

    start = time.time()

    # for saving index of mnn pairs in every update
    mnn_index_all = []

    # Start training
    for epoch in range(train_epoch):

        # Update every 10 epochs
        if epoch % 10 == 0 or epoch == 0:
            adata.obsm['ae_emb'] = AE_model.EncodeAll(adata.X, np.array(batch_onehot)).cpu().numpy()
            con_pairs = pairs_dataset(adata, mode='train', knn=10, alpha=.9, augment_set=['int'], exclude_fn=True, verbose=1)
            con_pairs.update_pos_nn_info()
            mnn_index = np.array(con_pairs.rand_pos_ind)
            knn_index_1 = np.array(con_pairs.rand_nn_ind1)
            knn_index_2 = np.array(con_pairs.pos_nn_ind)
            mnn_index_all.append(np.array(con_pairs.rand_pos_ind))
            con_pairs_loader = make_dataloader(adata, con_pairs, mnn_index,
                                               knn_index_1, knn_index_2, batch_size=batch_size)

        sum_con, sum_e, sum_mse = 0, 0, 0

        for data in con_pairs_loader:

            x = data[0].float().to(device)
            y = data[1].float().to(device)
            b = data[2].float().to(device)
            x1 = data[3].float().to(device)
            x2 = data[4].float().to(device)

            z, q, k, x_x = AE_model(x, b, x1, x2, y)

            loss_mse_x = MSE(x, x_x)
            loss_con = loss_fn(q, k)

            loss_e = con_w * loss_con + loss_mse_x

            sum_e += loss_e.item()
            sum_con += loss_con.item()
            sum_mse += loss_mse_x.item()

            # optimize AE
            optimizer_ae.zero_grad()
            loss_e.backward()
            optimizer_ae.step()

        print(
            'train epoch [{}/{}]. MSE loss:{:.4f}, Contrast loss:{:.4f}, total loss:{:.4f}'.format(
                epoch + 1, train_epoch, sum_mse / len(con_pairs_loader),
                sum_con / len(con_pairs_loader),
                sum_e / len(con_pairs_loader)))

    end = time.time()

    print(f'train use {end - start} seconds')

    ################################## Calculate metrics ##################################
    cyclone_AE_emb = AE_model.EncodeAll(adata.X, np.array(batch_onehot)).cpu().numpy()
    adata_l = sc.AnnData(cyclone_AE_emb)
    sc.pp.neighbors(adata_l)
    sc.tl.leiden(adata_l, resolution=0.17, random_state=0)
    y_pred = np.asarray(adata_l.obs['leiden'], dtype=int)
    n_clusters = len(np.unique(y_pred))
    print('Number of clusters identified by Leiden is {}'.format(n_clusters))


    nmi, ari = calculate_metric(y_pred, adata.obs['celltype'])
    adata.obsm['feat'] = cyclone_AE_emb
    BASW = calculate_BASW(adata)
    BER = calculate_ber(adata.obs['celltype'], y_pred, adata.obs['Batch'])
    print('Cluster : NMI= %.4f, ARI= %.4f,  BASW= %.4f, BER= %.4f,' % (
        nmi, ari, BASW, BER))

    end = time.time()
    run_time = end - start
    print(f'Total time: {end - start} seconds')

    ############################   Return results   #########################
    adata.obsm['cyclone_AE_emb'] = cyclone_AE_emb
    adata.obs['cyclone_AE_cluster'] = y_pred

    K = len(np.unique(y_pred))

    return adata, nmi, ari, K, run_time, BASW, BER, mnn_index_all
## Summary of the simulated data

For simulated data, we used the R package splatter  to generate a synthetic single-cell gene expression matrix with ground truth cell type labels and batch informations. First,  5000 genes and three batches with
1000, 1000 and 2000 cells were simulated using the following parameters: batch.facLoc=0.5, batch.facScale=0.5, de.facLoc=0.3, dropout.shape=−1, dropout.type=‘experiment’. In each batch, four
cell groups with equal probabilities (0.25) were set. Then, all cells from Group 4 were removed from Batch 1 and 2, so that Group 4 was exclusively present in Batch 3. The simulation data is available from 
https://drive.google.com/drive/folders/19DtVcatryAXm7tCB7bCQNeM_OUJBNAYf.

##Summary of the Bct-s data

The Bct-s (Mammary Epithelial Subset) dataset was developed in the Bct (Mammary Epithelial) dataset (cell number was 9288, gene number was 1222; 3 cell types, basal, luminal_mature, luminal_progenitor; 3 batches, vis, wal, spk) to remove the cells with the cell type luminal_mature in the batch ID of the spk, so that the cells with the cell type of luminal_mature in the dataset only exist in the batch vis and wal and become the batch-specific cell type. The Bct data is available fromhttps://doi.org/10.6084/m9.figshare.20499630.v2. The Bct-s data is available from https://drive.google.com/drive/folders/19DtVcatryAXm7tCB7bCQNeM_OUJBNAYf.

## Summary of all scRNA-seq datasets

|     Dataset     | No. of cells | No. of genes | No. of groups | No. of batches |
| :-------------: | :----------: | :----------: | :-----------: | :------------: |
|   Simulation    |     4000     |     5000     |       4       |       3        |
|       MCA       |     6954     |    15006     |      11       |       2        |
|      Bct-s      |     8245     |     1222     |       3       |       3        |
|   Human PBMC    |    15476     |    33694     |       9       |       2        |
| Human Pancreas  |    16382     |    19093     |      14       |       9        |
| Human Lung Cell |    32472     |    15148     |      17       |       16       |

## Download

All datasets can be downloaded from https://drive.google.com/drive/folders/1c33An3HNdJQhazoy_ky9E-lCc3a4y7fl or downloaded from the following links, in addition to the Simulation dataset and the Bct-s dataset, as these two datasets are generated.

|     Dataset     |                           Website                            |
| :-------------: | :----------------------------------------------------------: |
|       MCA       | (https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking/tree/master/Data) |
|   Human PBMC    | (https://github.com/JinmiaoChenLab/Batch-effect-removal-benchmarking/tree/master/Data) |
| Human Pancreas  |       https://doi.org/10.6084/m9.figshare.12420968.v8        |
| Human Lung Cell |       https://doi.org/10.6084/m9.figshare.12420968.v8        |

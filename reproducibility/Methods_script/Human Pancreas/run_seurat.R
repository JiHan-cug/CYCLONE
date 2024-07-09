rm(list=ls())
library(SeuratDisk)
library(Seurat)
library(stringr)
library(aricode)

Convert(paste0("../../data/Human Pancreas/human_pancreas_norm_complexBatch.h5ad"), dest="h5seurat",
        assay = "RNA",
        overwrite=F)
seurat_object <- LoadH5Seurat(paste0("../data/pancreas.h5seurat"))
seurat_object = UpdateSeuratObject(seurat_object)
simulate.list <- SplitObject(seurat_object, split.by = "tech")

simulate.list <- lapply(X = simulate.list, FUN = function(x) {
  x <- NormalizeData(x)
  return(x)
})

simulate.list <- lapply(X = simulate.list, FUN = function(x) {
  x <- FindVariableFeatures(x)
  return(x)
})

ptm <- proc.time()
simulate.anchors <- FindIntegrationAnchors(object.list = simulate.list, dims = 1:20)
simulate.combined <- IntegrateData(anchorset = simulate.anchors)
simulate.combined <- ScaleData(simulate.combined, verbose = FALSE)
simulate.combined <- RunPCA(simulate.combined, verbose = FALSE)
xpca = simulate.combined@reductions[["pca"]]@cell.embeddings
time = proc.time()-ptm
print(time)

simulate.combined <- FindNeighbors(simulate.combined)
simulate.combined <- FindClusters(simulate.combined, resolution = 0.5)
clust = simulate.combined@meta.data[["seurat_clusters"]]
write.csv(clust, paste0("pancreas_seurat_clust.csv"))

results = data.frame(xpca,
                     simulate.combined@meta.data[["celltype"]],
                     simulate.combined@meta.data[["tech"]]))
write.csv(results, paste0("pancreas_seurat_emb.csv"))

print(memory.profile())
print(memory.size(max=TRUE))



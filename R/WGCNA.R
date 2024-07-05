if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
# Check if WGCNA is installed
if (!requireNamespace("WGCNA", quietly = TRUE)) {
  print("installing WGCNA")
  BiocManager::install("WGCNA")
}
# Check if tidyverse is installed
if (!requireNamespace("tidyverse", quietly = TRUE)) {
  print('installing tidyverse')
  install.packages("tidyverse")
}

library('WGCNA')
library('tidyverse')
allowWGCNAThreads()
enableWGCNAThreads(8)

#' Create Blockwise Modules
#'
#' This function creates blockwise modules using the WGCNA package, suitable for network analysis of large datasets. It computes the topological overlap matrix (TOM) and identifies modules of highly correlated genes.
#'
#' @param df A dataframe where rows are samples and columns are features (e.g., gene expressions).
#' @param soft.power The soft-thresholding power to use in the adjacency function. This parameter can greatly affect the sensitivity to low correlations and the overall network connectivity.
#' @param merge.cut.height The height for merging modules in the dendrogram produced by hierarchical clustering. This parameter controls the granularity of the module detection, with lower values leading to more and smaller modules.
#'
#' @return A list containing the results of the blockwise module detection, including module labels for each feature, the feature network TOM, and the dendrogram of module clustering.
#'
#' @examples
#' data <- matrix(rnorm(1000), ncol=100)
#' result <- create_blockwise_modules(data, soft.power = 6, merge.cut.height = 0.25)
#' @export
#'
#' @importFrom WGCNA blockwiseModules
create_blockwise_modules <- function(df, soft.power, merge.cut.height){
    print('creating blockwise modules')
    temp_cor <- cor
    cor <- WGCNA::cor
    bwnet <- blockwiseModules(t(df),
                                maxBlockSize = 15000,
                                TOMType = "signed",
                                power = soft.power,
                                mergeCutHeight = merge.cut.height,
                                minModuleSize = 250,
                                numericLabels = FALSE,
                                randomSeed = 1234,
                                verbose = 3,
                                nThreads = 8)
    return(bwnet)
}
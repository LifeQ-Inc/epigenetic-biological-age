# Load required libraries and source scripts
library(dplyr)
source("enrichment.R")
source("dmp.R")

# Load wearable input data and epigenetic ages
wearable.inputs <- read.csv("../data/wearable_data.csv", row.names=1)
epigenetic.ages <- read.csv("../data/epigenetic_ages.csv", row.names=1)

# Load models and extract features vector
if (file.exists("../data/PCPhenoAgeAccel_model_performance.csv")){
  print("Loading model performance data")
  models <- read.csv("../data/PCPhenoAgeAccel_model_performance.csv")
  features_vector <- sapply(strsplit(gsub("\\[|\\]|'", "", models$Features), ", ")[[1]], function(x) gsub(" ", ".", x))
  wearable.inputs <- wearable.inputs %>% select(all_of(features_vector))
  print(paste("Wearable features to be used:", paste(colnames(wearable.inputs), collapse = ", ")))
} else {
  stop("Model performance data not found, please run the python model training script first")
}

# Load or calculate normalized beta values
if (file.exists("../data/normalizedbeta.RData")){
  print("Loading normalized beta values")
  load("../data/normalizedbeta.RData")
} else {
  beta <- load_raw_data("../data/epic_raw_data")
  save(beta, file = "../data/normalizedbeta.RData")
}

# Load or calculate DMP results for PCPhenoAge acceleration
if (file.exists("../data/PCPhenoAgeAccelDMP.RData")){
  print("Loading PCPhenoAgeAccel DMP results")
  load("../data/PCPhenoAgeAccelDMP.RData")
} else {
  print('Running DMP on entire methylation set for PCPhenoAge acceleration, this may take a while')
  PCPhenoAgeResid.dmp <- dmp_limma(beta = beta, features = epigenetic.ages$PCPhenoAge.acceleration, adjust = TRUE)
  save(PCPhenoAgeResid.dmp, file = "../data/PCPhenoAgeAccelDMP.RData")
}

print(paste("Number of DMPs for PCPhenoAge acceleration:", nrow(PCPhenoAgeResid.dmp)))

# Load or calculate DMP results for wearable features
if (file.exists("../data/WearableFeatsDMP.RData")){
  print("Loading wearable features DMP results")
  load("../data/WearableFeatsDMP.RData")
} else {
  print("Running DMP on entire methylation set for wearable features, this may take a while")
  wearable.feats.DMP <- dmp_limma(features = wearable.inputs, beta = beta, adjust = FALSE)
  wearable.feats.DMP_filtered <- subset(wearable.feats.DMP, adj_R_squared >= quantile(PCPhenoAgeResid.dmp$R_squared, 0.05))
  rownames(wearable.feats.DMP_filtered) <- wearable.feats.DMP_filtered$CpG_Site
  save(wearable.feats.DMP_filtered, file = "../data/WearableFeatsDMP.RData")
}

print(paste("Number of DMPs for wearable features:", nrow(wearable.feats.DMP_filtered)))

if (file.exists("../data/PCPhenoAgeResid_DMP_bwnet.RData")){
  print("Loading WGCNA results")
  load("../data/PCPhenoAgeResid_DMP_bwnet.RData")
} else {
  print("Running WGCNA on PCPhenoAge acceleration subset using wearable features, this may take a while")
  bwnet <- create_blockwise_modules(t(beta[rownames(PCPhenoAgeResid.dmp), ]), 12, 0.15)
  save(bwnet, file = "../data/PCPhenoAgeResid_DMP_46_samples_bwnet.RData")
}

# Extract module sites and perform gene enrichment analysis and save the output to csv
print("extracting module sites")
moduleColors <- bwnet$colors 
module.df.list <- list()
for (i in unique(moduleColors)) {
  if (i != "grey") {
    #TODO this df needs to be from the dmp calculated above. 
    df.colors <- PCPhenoAgeResid.dmp[moduleColors == i,]
    # common.module.sites <- df.colors[df.colors %in% wearable.feats.DMP_filtered$CpG_Site]
    rownames(df.colors) <- df.colors$CpG_Site
    module.df.list[[i]] <- df.colors
    print(i)
    print(nrow(df.colors))
  }
}
print('running enrichment on:')
print(names(module.df.list))

# add the wearable DMP df to this list 
module.df.list[["wearable"]] <- wearable.feats.DMP_filtered

enrichment.results <- performGeneEnrichment_list(module.df.list, dbs, anno)

for (name in names(enrichment.results)) {
  if (name != 'wearable') {
    # Filter the dataframe based on terms in the 'wearable' dataframe
    filtered_df <- enrichment.results[[name]][enrichment.results[[name]]$Term %in% enrichment.results[['wearable']]$Term, ]
    # check if the df is empty, if it is don't write to csv
    if (nrow(filtered_df) > 0) {
      write.csv(filtered_df, paste0(name, ".csv"), row.names = FALSE)
    }
  } else {
    # Assuming you want to write the 'wearable' dataframe as is
    write.csv(enrichment.results[[name]], paste0(name, ".csv"), row.names = FALSE)
  }
}

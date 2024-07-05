library("enrichR")
library("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")

anno <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
anno$UCSC_RefGene_Name <- gsub(";.*", "", anno$UCSC_RefGene_Name)
  
# Set Enrichr site and indicate website is live
setEnrichrSite("Enrichr")
websiteLive <- TRUE
  
# Define databases for enrichment
dbs <- c('GO_Biological_Process_2023','GO_Cellular_Component_2023','GO_Molecular_Function_2023','Reactome_2022','KEGG_2021_Human', "Human_Gene_Atlas")

performGeneEnrichment <- function(data_frames, dbs, anno) {
  # Get and prepare annotation data
  
  # Initialize an empty list to store merge results
  merge_results <- list()
  
  # Loop through the list of data frames, merge with anno, and filter empty rows
  for (name in names(data_frames)) {
    print(name)
    # Convert row names of the current data frame to a regular column for merging
    data_frames[[name]]$Row.names <- rownames(data_frames[[name]])
    # Ensure 'anno' has 'Row.names' as a regular column or convert its row names
   
    print('merging')
    merged <- merge(data_frames[[name]], anno[, c("Row.names", "UCSC_RefGene_Name")], by = 'Row.names')
    merged <- merged[!apply(merged, 1, function(x) any(x == "")),]
    merge_results[[name]] <- merged
    print('done merging')
  }
  
  enrichment_results <- list()
  
  # Perform enrichment if the website is live
  if (websiteLive) {
    for (name in names(merge_results)) {
      print('running enrichment')
      enrichment_results[[name]] <- enrichr(c(merge_results[[name]]$UCSC_RefGene_Name), dbs)
    }
  }
  
  # Extract and filter enrichment results based on Adjusted P-value
  result_list <- list()
  for (name in names(enrichment_results)) {
    result_list[[name]] <- enrichment_results[[name]]$KEGG_2021_Human %>% filter(Adjusted.P.value < 0.05)
  }
  
  return(result_list)
}

#' Perform Gene Enrichment Analysis on Multiple Data Frames
#'
#' This function performs gene enrichment analysis on multiple data frames using specified databases and annotations.
#'
#' @param data_frames A list of data frames containing gene data.
#' @param dbs A vector of database names to be used for enrichment analysis.
#' @param anno A data frame containing gene annotations.
#'
#' @return A list of data frames, each containing filtered enrichment results based on Adjusted P-value for each input data frame.
#'
#' @examples
#' result <- performGeneEnrichment_list(data_frames, dbs, anno)
#'
#' @importFrom dplyr filter
performGeneEnrichment_list <- function(data_frames, dbs, anno) {
  # Ensure 'anno' has 'Row.names' as a regular column or convert its row names
  if (!"Row.names" %in% colnames(anno)) {
    anno$Row.names <- rownames(anno)
  }
  
  # Initialize an empty list to store filtered annotation results
  filtered_anno_results <- list()
  
  # Loop through the list of data frames, filter anno based on row names
  for (name in names(data_frames)) {
    print(name)
    # Get the row names of the current data frame
    current_df_row_names <- rownames(data_frames[[name]])
    # Filter 'anno' to only include rows where 'UCSC_RefGene_Name' matches the row names of the current dataframe
    filtered_anno <- anno[rownames(anno) %in% current_df_row_names, ]
    filtered_anno_results[[name]] <- filtered_anno
  }
  
  enrichment_results <- list()
  
  # Perform enrichment if the website is live
  if (websiteLive) {
    for (name in names(filtered_anno_results)) {
      print('running enrichment')
      # Use the 'UCSC_RefGene_Name' column from the filtered anno for enrichment
      enrichment_results[[name]] <- enrichr(c(filtered_anno_results[[name]]$UCSC_RefGene_Name), dbs)
    }
  }
  
  # Extract and filter enrichment results based on Adjusted P-value
  result_list <- list()
  for (name in names(enrichment_results)) {
    result_list[[name]] <- enrichment_results[[name]]$KEGG_2021_Human %>% filter(Adjusted.P.value < 0.05)
  }
  
  return(result_list)
}
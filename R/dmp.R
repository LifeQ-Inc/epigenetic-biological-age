library("tidyverse")
library("limma")
library("ChAMP")
library("glmnet")

#' Load and Normalize Raw Data
#'
#' This function loads raw methylation data from a specified file path and performs normalization using the ChAMP package. It is specifically designed for EPIC array data.
#'
#' @param path The file path to the raw data.
#'
#' @return A matrix of normalized beta values, where rows represent CpG sites and columns represent samples.
#'
#' @examples
#' normalized_beta <- load_raw_data("path/to/your/data.csv")
#'
#' @importFrom ChAMP champ.load champ.norm
load_raw_data <- function(path){
  myLoad <- champ.load(path, arraytype = "EPIC")
  beta <- champ.norm(myLoad$beta)

  return(beta)
}

#' Differential Methylation Positions Analysis
#'
#' This function performs differential methylation positions (DMP) analysis using the ChAMP package. It identifies CpG sites that are differentially methylated between different phenotypic groups.
#'
#' @param beta_values A matrix of beta values where rows are CpG sites and columns are samples.
#' @param phenotype A vector indicating the phenotype of each sample, used to divide samples into groups for comparison.
#'
#' @return The function currently does not explicitly return a value. To enhance functionality, consider modifying the function to return the `champ.dmp` result for further analysis or visualization.
#'
#' @examples
#' beta_values <- matrix(rnorm(1000), nrow=100, ncol=10)
#' phenotype <- rep(c("Control", "Treatment"), each=5)
#' dmp_result <- dmp(beta_values, phenotype)
#'
#' @importFrom ChAMP champ.DMP
dmp <- function(beta_values, phenotype){
    print('running dmp')
    champ.dmp <- champ.DMP(
        beta = beta_values,
        pheno = phenotype,
        arraytype = 'EPIC',
        adjust.method = 'fdr',
        adjPVal = 0.05
    )
}

#' Parse Model Summaries
#'
#' This function extracts F-statistic, p-value, R-squared, and adjusted R-squared from a list of model summaries and adjusts p-values using FDR.
#'
#' @param model_summaries A list of model summaries.
#'
#' @return A data frame containing the F-statistic, F p-value, R-squared, adjusted R-squared, and adjusted p-value for each model summary.
#'
#' @examples
#' parsed_summaries <- parse_model_summaries(model_summaries)
#'
#' @importFrom dplyr mutate
#' @importFrom purrr map_dfr
#' @importFrom tibble tibble
parse_model_summaries <- function(model_summaries) {
  # Function to extract F-statistic and p-value from a model summary
  extract_stats <- function(model_summary) {
    f_statistic <- model_summary$fstatistic[1]
    f_p_value <- pf(f_statistic, model_summary$fstatistic[2], model_summary$fstatistic[3], lower.tail = FALSE)
    r_squared <- model_summary$r.squared
    adj_r_squared <- model_summary$adj.r.squared
    
    tibble(F_statistic = f_statistic, F_p_value = f_p_value, R_squared = r_squared, adj_R_squared = adj_r_squared)
  }
  
  # Use purrr::map_dfr() to apply the function to each model summary and combine the results into a data frame
  results <- map_dfr(model_summaries, extract_stats, .id = "CpG_Site")
  
  results <- results %>%
    mutate(CpG_Site = sub("Response ", "", CpG_Site))
  
  results$adj.p.value = p.adjust(results$F_p_value, method = "fdr")
  
  return(results)
}

#' Differential Methylation Analysis using Limma
#'
#' This function prepares data for differential methylation analysis using the Limma package, ensuring that feature and beta value dimensions match.
#'
#' @param features A matrix or data frame of features.
#' @param beta A matrix of beta values where rows represent CpG sites and columns represent samples.
#' @param adjust A logical value indicating whether to adjust p-values.
#'
#' @return The function currently does not return a value. It is recommended to modify the function to return the result of the analysis.
#'
#' @examples
#' dmp_limma_result <- dmp_limma(features, beta, TRUE)
#'
#' @importFrom stats scale
dmp_limma <- function(features, beta, adjust){
  # Scale the features and convert to a data frame
  features <- as.data.frame(scale(features))
  
  # Check if the number of samples in features matches the number of samples in beta
  if (nrow(features) != ncol(beta)) {
    stop("Number of samples in features does not match number of samples in beta")
  }
  
  # beta value transformation
  beta <- log2((beta + 0.01) / (1 - beta + 0.01))
  
  design <- model.matrix(~  ., data = features)
  fit1 <- lmFit(beta, design)
  
  fit2 <- eBayes(fit1)
  if (adjust == TRUE){
    DMP <- topTable(fit2, coef = NULL, number=nrow(beta), adjust.method = "fdr", p.value = 0.05)
  }
  else 
    DMP <- topTable(fit2, coef = NULL, number=nrow(beta))
  
  model <- lm(t(beta) ~., data = features)
  model_summary <- summary(model)
  model_summary <- parse_model_summaries(model_summary)
  
  DMP$CpG_Site <- rownames(DMP)
  
  dmp.df <- merge(DMP, model_summary, by = "CpG_Site")
  
  return(dmp.df)
}
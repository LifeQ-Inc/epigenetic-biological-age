import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_feats(df):
    """
    Processes the features of a DataFrame by removing highly correlated features.

    This function calculates the correlation matrix of the DataFrame, identifies features that are highly correlated 
    (correlation > 0.9), and removes these features from the DataFrame. It logs the number of original features, 
    the number of features to drop, the number of processed features, and the features to drop along with their 
    correlated features.

    Parameters:
    df (DataFrame): The DataFrame containing the features to process.

    Returns:
    DataFrame: The processed DataFrame with highly correlated features removed.
    """

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    highly_correlated = {column: upper.index[upper[column] > 0.9].tolist() for column in to_drop}

    processed_features = df.drop(columns=to_drop, errors='ignore')
    # processed_sleep_features = sleep_features_all.copy()

    logging.info(f"Number of features: {len(df.columns)}")
    logging.info(f"Number of features to drop: {len(to_drop)}")
    logging.info(f"Number of processed features: {len(processed_features.columns)}")
    logging.info("Features to drop and their correlated features:")
    for feature, correlated_features in highly_correlated.items():
        logging.info(f"{feature}: {correlated_features}")
    logging.info(f"features to drop {to_drop}")

    return processed_features

def perform_pca(df, n):
    """
    Performs Principal Component Analysis (PCA) on a DataFrame and returns the PCA object and the transformed DataFrame.

    This function scales the features of the DataFrame, performs PCA to reduce the dimensionality of the data, 
    and creates a new DataFrame with the principal components. It also identifies outliers based on a threshold 
    and logs the indices of these outliers.

    Parameters:
    df (DataFrame): The DataFrame on which to perform PCA.
    n (int): The number of principal components to retain.

    Returns:
    pca (PCA object): The PCA object that was fit to the data.
    pca_df (DataFrame): The DataFrame of principal components.
    """

    pca_df = pd.DataFrame(index=df.index)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n)
    pca1 = pca.fit_transform(features_scaled)

    for i in range(n):
        pca_df['PCA' + str(i+1)] = pca1[:, i]

    threshold = 6
    outlier_condition = (pca_df['PCA2'] > threshold) | (pca_df['PCA1'] > threshold)
    index_to_remove = pca_df[outlier_condition].index
    logging.info(index_to_remove)

    return pca, pca_df

def scree_plot(pca):
    """
    Plots a scree plot based on the explained variance ratio of a PCA object.

    This function calculates the explained variance ratio of the PCA object, logs it, and then creates a scree plot.
    The scree plot is a bar plot with the principal components on the x-axis and the proportion of variance explained 
    on the y-axis. A horizontal dashed line is drawn at y=0.15 for reference.

    Parameters:
    pca (PCA object): The PCA object from which to calculate the explained variance ratio.
    """

    explained_variance = pca.explained_variance_ratio_
    logging.info(explained_variance)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance)+1), explained_variance)
    plt.xticks(range(1, len(explained_variance)+1))  # Set locations and labels
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.title('Scree Plot')
    plt.hlines(0.15, 0, len(explained_variance)+1, colors='r', linestyles='dashed')
    plt.tight_layout()
    plt.savefig('plots/pca_scree_plot.svg')

def eigenvector_plot(pca, processed_features):
    """
    Plots the eigenvectors of the first two principal components of a PCA object.

    This function creates a DataFrame of the PCA components, transposes it to get the eigenvectors, 
    and then plots a bar plot of the eigenvectors of the first two principal components. 
    The x-axis represents the features and the y-axis represents the eigenvectors. 
    Horizontal dashed lines are drawn at y=-0.2 and y=0.2 for reference.

    Parameters:
    pca (PCA object): The PCA object from which to extract the components.
    processed_features (DataFrame): The DataFrame of processed features.
    """
    df_comp = pd.DataFrame(pca.components_[0:2], columns=processed_features.columns)
    eigenvectors = pd.DataFrame(df_comp.iloc[0:2].T)

    plt.figure(figsize=(12, 6))
    plt.bar(eigenvectors.index, eigenvectors[0], label='PCA1')
    plt.bar(eigenvectors.index, eigenvectors[1], label='PCA2', alpha=0.5)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Eigenvector')
    plt.title('PCA1 and PCA2 Eigenvectors')
    plt.axhline(y=-0.2, color='r', linestyle='--')
    plt.axhline(y=0.2, color='r', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/pca_eigenvector.svg')

def calc_loadings(pca, processed_features):
    """
    Calculates the component loadings of the first two principal components of a PCA object.

    This function calculates the component loadings, creates a DataFrame for the loadings, 
    and checks if the sum of the squared weights equals the eigenvalues of the components. 
    If the sum of the squared weights does not equal the eigenvalue, it prints False.

    Parameters:
    pca (PCA object): The PCA object from which to extract the components.
    processed_features (DataFrame): The DataFrame of processed features.

    Returns:
    loadings_df (DataFrame): The DataFrame of component loadings.
    """

    loadings = pca.components_[0:2].T * np.sqrt(pca.explained_variance_[0:2])
    loadings_df = pd.DataFrame(loadings, columns=['PCA1', 'PCA2'], index=processed_features.columns)

    if np.sum(loadings_df['PCA1'] ** 2).round(2) != pca.explained_variance_[0].round(2):
        print(False)
    if np.sum(loadings_df['PCA2'] ** 2).round(2) != pca.explained_variance_[1].round(2):
        print(False)

    return loadings_df

def plot_loadings(loadings_df):
    """
    Plots the loadings of the first two principal components.

    This function creates a bar plot of the loadings of the first two principal components. 
    The x-axis represents the features and the y-axis represents the loadings. 
    Horizontal dashed lines are drawn at y=-0.75 and y=0.75 for reference. 
    The plot is saved as 'pca_loadings.png' in the 'plots' directory.

    Parameters:
    loadings_df (DataFrame): The DataFrame of component loadings.
    """

    plt.figure(figsize=(12, 6))
    plt.bar(loadings_df.index, loadings_df['PCA1'], label='PCA1')
    plt.bar(loadings_df.index, loadings_df['PCA2'], label='PCA2', alpha=0.5)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Loading')
    plt.title('PCA1 and PCA2 Loadings')
    plt.axhline(y=-0.75, color='r', linestyle='--')
    plt.axhline(y=0.75, color='r', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/pca_loadings.svg')

def run_pca_process(features, n, drop):
    if drop:
        processed_features = process_feats(features)
    else:
        processed_features = features.copy()

    pca, pca_df = perform_pca(processed_features, n)

    scree_plot(pca)
    eigenvector_plot(pca, processed_features)

    loadings_df = calc_loadings(pca, processed_features)

    plot_loadings(loadings_df)

    importance_threshold = 0.70
    important_features_pca1 = loadings_df[(loadings_df['PCA1'] > importance_threshold) | (loadings_df['PCA1'] < -importance_threshold)]
    important_features_pca2 = loadings_df[(loadings_df['PCA2'] > importance_threshold) | (loadings_df['PCA2'] < -importance_threshold)]

    logging.info(f"Important features for PCA1:\n {important_features_pca1}")
    logging.info(f"Important features for PCA2:\n {important_features_pca2}")

    return pca_df

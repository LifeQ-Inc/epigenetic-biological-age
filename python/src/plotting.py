import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

def plot_clustermap(corrs, p_values_corrected, plot_row_colors, plot_column_colors, q, p, row_colors=None, row_features=None, column_colors=None, col_features=None, name=None):
    # Convert correlations to numeric
    correlations = corrs.apply(pd.to_numeric)

    if plot_row_colors:
        row_colors_series = pd.Series(row_features).map(row_colors)
    else:
        row_colors_series = None

    if plot_column_colors:

        col_colors_series = correlations.columns.map(column_colors)
    else:
        col_colors_series = None

    # Create a mask for significant correlations
    mask = p_values_corrected < 0.05

    # Create a DataFrame for the annotations
    annotations = correlations.astype(str).applymap(lambda x: f"{float(x):.2f}")
    if q:
        annotations[mask] = annotations[mask] + '*'
    elif p:
        annotations[mask] = annotations[mask] + '#'

    g = sns.clustermap(correlations,
                    cmap='coolwarm', 
                    figsize=(10, 8), 
                    col_colors=col_colors_series, 
                    row_colors=row_colors_series,
                    annot= annotations,
                    fmt='s')

    if plot_row_colors:
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in row_colors.items()]
        g.cax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(5, 1))

    plt.savefig(f"plots/{name}_clustermap.svg", dpi = 300)

def plot_feature_distributions(feats, features_dict, feature_colors):
    num_cols = feats.shape[1]
    num_rows = num_cols // 2 if num_cols % 2 == 0 else num_cols // 2 + 1
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, num_rows*5))
    axs = axs.flatten()

    for i, col in enumerate(feats.columns):
        # calculate the minimum, median, and maximum
        min_val = feats[col].min()
        median_val = feats[col].median()
        max_val = feats[col].max()
        # plot the histogram
        sns.histplot(feats[col], kde=True, ax=axs[i], color = feature_colors[features_dict[col]])
        # plot the minimum, median, and maximum
        for val in [min_val, median_val, max_val]:
            axs[i].axvline(val, color = 'black', linestyle = 'dashed', linewidth = 1.5, alpha = 0.7)
        # plot minimum, median, and maximum in the legend
        axs[i].legend([f'min: {min_val:.2f}', f'median: {median_val:.2f}', f'max: {max_val:.2f}'])
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.svg')

def plot_target_distribution(target_train, target_test):
    """
    Plots the distribution of target variables 'ChronAge' and 'PCPhenoAgeResid' for both training and testing datasets.

    Parameters:
    target_train (DataFrame): The training dataset containing the target variables.
    target_test (DataFrame): The testing dataset containing the target variables.

    Returns:
    None. A plot is saved as 'plots/model_distributions.png'.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    median_train_chron = round(target_train['ChronAge'].median(), 2)
    q25_train_chron = round(target_train['ChronAge'].quantile(0.25), 2)
    q75_train_chron = round(target_train['ChronAge'].quantile(0.75), 2)

    sns.histplot(target_train['ChronAge'], kde=True, color='blue', label='Training', ax=axes[0, 0])
    axes[0, 0].axvline(median_train_chron, color='r', linestyle='--', label=f'Median: {median_train_chron}')
    axes[0, 0].axvline(q25_train_chron, color='r', linestyle=':', label=f'Q1: {q25_train_chron}')
    axes[0, 0].axvline(q75_train_chron, color='r', linestyle=':', label=f'Q3: {q75_train_chron}')
    axes[0, 0].set_title('ChronAge - Training')
    axes[0, 0].legend()

    median_test_chron = round(target_test['ChronAge'].median(), 2)
    q25_test_chron = round(target_test['ChronAge'].quantile(0.25), 2)
    q75_test_chron = round(target_test['ChronAge'].quantile(0.75), 2)

    sns.histplot(target_test['ChronAge'], kde=True, color='green', label='Testing', ax=axes[0, 1])
    axes[0, 1].axvline(median_test_chron, color='r', linestyle='--', label=f'Median: {median_test_chron}')
    axes[0, 1].axvline(q25_test_chron, color='r', linestyle=':', label=f'Q1: {q25_test_chron}')
    axes[0, 1].axvline(q75_test_chron, color='r', linestyle=':', label=f'Q3: {q75_test_chron}')
    axes[0, 1].set_title('ChronAge - Testing')
    axes[0, 1].legend()

    median_train_pc = round(target_train['PCPhenoAgeResid'].median(), 2)
    q25_train_pc = round(target_train['PCPhenoAgeResid'].quantile(0.25), 2)
    q75_train_pc = round(target_train['PCPhenoAgeResid'].quantile(0.75), 2)

    sns.histplot(target_train['PCPhenoAgeResid'], kde=True, color='blue', label='Training', ax=axes[1, 0])
    axes[1, 0].axvline(median_train_pc, color='r', linestyle='--', label=f'Median: {median_train_pc}')
    axes[1, 0].axvline(q25_train_pc, color='r', linestyle=':', label=f'Q1: {q25_train_pc}')
    axes[1, 0].axvline(q75_train_pc, color='r', linestyle=':', label=f'Q3: {q75_train_pc}')
    axes[1, 0].set_title('PCPhenoAgeResid - Training')
    axes[1, 0].legend()

    median_test_pc = round(target_test['PCPhenoAgeResid'].median(), 2)
    q25_test_pc = round(target_test['PCPhenoAgeResid'].quantile(0.25), 2)
    q75_test_pc = round(target_test['PCPhenoAgeResid'].quantile(0.75), 2)

    sns.histplot(target_test['PCPhenoAgeResid'], kde=True, color='green', label='Testing', ax=axes[1, 1])
    axes[1, 1].axvline(median_test_pc, color='r', linestyle='--', label=f'Median: {median_test_pc}')
    axes[1, 1].axvline(q25_test_pc, color='r', linestyle=':', label=f'Q1: {q25_test_pc}')
    axes[1, 1].axvline(q75_test_pc, color='r', linestyle=':', label=f'Q3: {q75_test_pc}')
    axes[1, 1].set_title('PCPhenoAgeResid - Testing')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('plots/model_distributions.svg')

def plot_model_predictions_scatter_and_distribution(target_values, target):
    """
    Plots scatter plots of actual vs predicted target values for both training and testing datasets, 
    and a histogram of the distribution of predicted and actual target values.

    Parameters:
    target_values (DataFrame): The dataset containing the actual and predicted target values, and a 'Dataset' column indicating whether each row is from the 'Train' or 'Test' dataset.
    target (str): The name of the target variable.
    """

    train_data = target_values[target_values['Dataset'] == 'Train']
    test_data = target_values[target_values['Dataset'] == 'Test']

    fig, axs = plt.subplots(2, 1, figsize=(6, 12))

    sns.regplot(x=train_data[target],
                y=train_data[f'predicted_{target}'], 
                color='royalblue',
                label='Train',
                scatter_kws={'s': 15},
                ci= None,
                line_kws = {'linestyle': "--"},
                ax=axs[0])

    sns.regplot(x=test_data[target],
                y=test_data[f'predicted_{target}'], 
                color='darkorange',
                label='Test',
                scatter_kws={'s': 15},
                ci = None,
                line_kws={'linestyle': '--'},
                ax=axs[0])

    axs[0].set_xlabel(f'Actual {target}')
    axs[0].set_ylabel('Predicted {target}')
    axs[0].legend()

    sns.histplot(target_values[target], alpha = 0.5, kde=True, label='Actual', color='blue', ax=axs[1])
    sns.histplot(target_values[f'predicted_{target}'], alpha = 0.5, kde= True,  label='Predicted', color='r', ax=axs[1])
    axs[1].legend(loc='upper left')
    axs[1].set_title('')
    axs[1].set_xlabel(f'Distribution of predicted and actual {target}')

    plt.savefig('plots/model_predictions_scatter_and_distribution.svg')

import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.savefig('plots/model_distributions.png')

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

    plt.savefig('plots/model_predictions_scatter_and_distribution.png')

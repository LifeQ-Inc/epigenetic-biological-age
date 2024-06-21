import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def extract_feature_importance(model, features, features_train, features_test, full_dataset=False):
    """
    Extracts the feature importance from a given model using SHAP (SHapley Additive exPlanations).

    This function creates a SHAP explainer based on the type of the input model. For tree-based models (RandomForestClassifier, 
    GradientBoostingClassifier, DecisionTreeClassifier), it uses the TreeExplainer. For linear models (LogisticRegression, SVC, 
    LinearRegression, Lasso, Ridge, ElasticNet), it uses the KernelExplainer. 

    The function then calculates the SHAP values for the training and test features. The SHAP values represent the contribution 
    of each feature to the prediction for each sample. A positive SHAP value indicates that the feature increases the prediction, 
    while a negative SHAP value indicates that the feature decreases the prediction.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained model for which to calculate feature importance.
    features_train (pd.DataFrame or np.array): The training features.
    features_test (pd.DataFrame or np.array): The test features.
    target (pd.Series or np.array): The target variable. Not used in the function.

    Returns:
    shap_values_train (np.array): The SHAP values for the training features.
    shap_values_test (np.array): The SHAP values for the test features.

    Raises:
    ValueError: If the model type is not supported.
    """
    
    if full_dataset:
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (LogisticRegression, SVC)):
            explainer = shap.KernelExplainer(model.predict, features)
        elif isinstance(model, (LinearRegression, Lasso, Ridge, ElasticNet)):
            explainer = shap.KernelExplainer(model.predict, features)
        else:
            raise ValueError(f"Model type {type(model)} not supported")
        shap_values = explainer.shap_values(features)
        return shap_values
    
    else:
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (LogisticRegression, SVC)):
            explainer = shap.KernelExplainer(model.predict, features_train)
        elif isinstance(model, (LinearRegression, Lasso, Ridge, ElasticNet)):
            explainer = shap.KernelExplainer(model.predict, features_train)
        else:
            raise ValueError(f"Model type {type(model)} not supported")
        shap_values_test = explainer.shap_values(features_test)
        shap_values_train = explainer.shap_values(features_train)

    return shap_values_train, shap_values_test

def plot_shap_violin_plot(importance, selected_features, features_scaled, features, model):
    """
    Plots a SHAP violin plot and saves it to a file.

    This function first calculates the absolute importance of features and sorts them in descending order.
    It then calculates the SHAP values for the given model and features.
    Finally, it plots a SHAP violin plot and saves it as 'plots/shap_summary.png'.

    Parameters:
    importance (array-like): The importance of the features.
    selected_features (array-like): The features selected for the model.
    features_scaled (DataFrame): The scaled features.
    features (DataFrame): The original features.
    model (model object): The trained model for which to calculate SHAP values.
    """
    
    feature_importance = pd.DataFrame(importance, index=selected_features, columns=['importance'])
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)

    shap_values = extract_feature_importance(model, features = features_scaled, features_test=None, features_train=None, full_dataset=True)
    plt.figure(figsize=(10,10))
    shap.summary_plot(shap_values, features[selected_features], plot_type="violin")
    plt.savefig('plots/shap_summary.svg')


def individual_feature_contribution_plot(features_scaled, selected_features, feature_importance, target_values, target, participant):
    """
    Plots the contribution of each feature to the predicted age acceleration for a specific participant.

    Parameters:
    features_scaled (numpy.ndarray): The scaled features.
    selected_features (list): The names of the selected features.
    feature_importance (dict): A dictionary with feature names as keys and their importance as values.
    target_values (pandas.DataFrame): A DataFrame containing the target values.
    participant (int): The index of the participant.

    The function creates a horizontal bar plot where each bar represents a feature and its contribution to the 
    predicted age acceleration. The contributions are color-coded based on whether they are positive (red) or 
    negative (blue). The predicted age acceleration for the participant is shown as a vertical dashed line.

    The plot is saved as 'plots/individual_feature_contribution.png'.
    """

    features_scaled_df = pd.DataFrame(features_scaled, columns=selected_features)
    column_order = ['BPI', 'SAR', 'normalized crest time', 'SEVR', 'ED', 'VO2max', 'LASI', 'RHR']
    features_scaled_df = features_scaled_df[column_order]
    individual_scaled = features_scaled_df.iloc[participant]
    individual_contributions = feature_importance['importance'] * individual_scaled
    individual_contributions = individual_contributions[features_scaled_df.columns]

    fig, ax = plt.subplots()
    bottom = 0

    for index, row in individual_contributions.items():
        color = 'red' if row > 0 else 'deepskyblue'
        alpha = 0.7 if row > 0 else 0.6 
        ax.barh(index, row, left=bottom, color=color, alpha = alpha)
        bottom += row

    predicted = target_values.iloc[participant][f'predicted_{target}']  
    ax.axvline(predicted, color='black', linestyle='--', alpha = 0.8)
    ax.text(predicted + 0.15, ax.get_ylim()[1] + 0.1, round(predicted, 2), color='black', ha='right', fontsize=12)
    ax.set_xlabel(f'Contribution to predicted {target} (years)', fontsize = 12)
    ax.set_title(f'Impact of wearable features on prediction of {target}', y = 1.05)
    ax.tick_params(axis='y', labelsize='large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig('plots/individual_feature_contribution.svg')
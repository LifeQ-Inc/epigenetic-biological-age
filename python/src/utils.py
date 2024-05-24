import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data():
    """
    Read data from a csv file stored in the data directory. The data needs to be a csv file called wearable.csv

    params: Path to csv file
    return: wearable_feats - dataframe of wearable features
    return: age_df - dataframe of the biological ages
    return: age_groups - dataframe of the age groups
    """

    df = pd.read_csv('data/wearable_data.csv')
    df.dropna(inplace=True)
    df.set_index('Sample_Name', inplace=True)

    if 'PCPhenoAge' in df.columns:
        age_df = df[['ChronAge','PCHorvath1', 'PCHorvath2', 'PCHannum', 'PCPhenoAge', 'PCGrimAge','ChronAge','PCHorvath1Resid', 'PCHorvath2Resid', 'PCHannumResid','PCPhenoAgeResid', 'PCGrimAgeResid']]
    elif os.path.exists('data/epigenetic_ages.csv'):
        logging.info('No epigenetic age data, reading from separate file')
        age_df = pd.read_csv('data/epigenetic_ages.csv')
        age_df.set_index('Sample_Name', inplace=True)
    else:
        logging.info('No epigenetic age data, please process epigenetic ages first')    
        age_df = None

    features = df[['ChronAge','VO2max', 'normalized crest time', 'normalized_crest_time_sd', 'SEVR', 'SEVR_sd', 'SAR', 'SAR_sd', 'LASI', 'LASI_sd', 'BPI', 'BPI_sd', 'ED', 'ED_sd', 'RHR', 
                'weekly light active hours', 'weekly moderate active hours', 'weekly vigorous active hours', 'daily sedentary hours','momentum_mean', 'totalSleepTime_all_mean', 'SQS_mean', 
                'total_rem_mean', 'percentage_rem_mean', 'total_deep_mean', 'percentage_deep_mean', 'total_light_mean', 'percentage_light_mean', 'waso_duration_mean', 'sleep_efficiency_mean',
                'momentum_cv', 'totalSleepTime_all_cv', 'total_rem_cv','percentage_rem_cv', 'total_deep_cv', 'percentage_deep_cv', 'total_light_cv', 'percentage_light_cv', 'waso_duration_cv',
                'sleep_efficiency_cv','SQS_cv']]
    sleep_features = df[['totalSleepTime_all_mean', 'SQS_mean', 'total_rem_mean', 'percentage_rem_mean', 'total_deep_mean', 'percentage_deep_mean', 'total_light_mean', 'percentage_light_mean', 'waso_duration_mean', 'sleep_efficiency_mean',
                'totalSleepTime_all_cv', 'total_rem_cv','percentage_rem_cv', 'total_deep_cv', 'percentage_deep_cv', 'total_light_cv', 'percentage_light_cv', 'waso_duration_cv', 'sleep_efficiency_cv','SQS_cv']]

    return df, features, age_df, sleep_features

def prepare_data(features, targets,  test_set_size, random_seed,stratification = True):
    """
    Prepares data for machine learning by performing train-test split, stratification, and scaling.
    The function first calculates the median of 'ChronAge' and creates a 'stratify_key' based on the stratification_option.
    It then splits the data into a training set and a test set, with the test set size specified by test_set_size.
    If stratification is True, the data is stratified based on the 'stratify_key'.
    The function then scales the feature data using StandardScaler from sklearn, which standardizes features by removing the mean and scaling to unit variance.
    The scaler is fitted on the training data and is then used to transform both the training and test data.

    Parameters:
    features (DataFrame): The DataFrame containing the feature variables.
    targets (DataFrame): The DataFrame containing the target variables.
    test_set_size (float): The proportion of the dataset to include in the test split (between 0.0 and 1.0).
    random_seed (int): The seed for the random number generator.
    stratification (bool): If True, stratification is performed based on 'PCPhenoAgeResid_bins' and 'ChronAge'.

    Returns:
    features_train (DataFrame): The features for the training set, scaled to have zero mean and unit variance.
    features_test (DataFrame): The features for the test set, scaled to have zero mean and unit variance.
    target_train (DataFrame): The targets for the training set.
    target_test (DataFrame): The targets for the test set.
    """

    median_chron_age = targets['ChronAge'].median()

    if stratification:
        bins = [-np.inf, -4, 0, 4, np.inf]
        labels = [0, 1, 2, 3]
        targets['PCPhenoAgeResid_bins'] = pd.cut(targets['PCPhenoAgeResid'], bins=bins, labels=labels)

        targets['stratify_key'] = targets['PCPhenoAgeResid_bins'].astype(str) + '-' + np.where(targets['ChronAge'] > median_chron_age, 'A', 'B')
    
    if stratification:
        features_train, features_test, target_train, target_test = train_test_split(features, targets, test_size= test_set_size, random_state= random_seed, stratify=targets['stratify_key'])
    else:
        features_train, features_test, target_train, target_test = train_test_split(features, targets, test_size= test_set_size, random_state= random_seed)

    logging.info(f'training group {len(features_train)}')
    logging.info(f'test group {len(features_test)}')

    return features_train, features_test, target_train, target_test

def extract_targets(df, target_variable):
    """
    Extracts the target variables from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the target variables.

    Returns:
    targets (pd.DataFrame): The DataFrame containing the target variables.
    """
    targets = df[target_variable]
    return targets

def extract_model_info(model_info):
    """
    Extracts the model, input features, and scale from the model_info dictionary.

    Returns:
    model (sklearn.base.BaseEstimator): The trained model.
    input_features (list): The list of input features used to train the model.
    scale (StandardScaler): The StandardScaler object used to scale the input features.
    """
    model = model_info['model']
    input_features = model_info['model_input_features']
    scale = model_info['scale']
    target = model_info['target']

    return model, input_features, scale, target
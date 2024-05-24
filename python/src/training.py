import logging
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import time
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def get_linear_models():
    """
    Returns a dictionary of linear models and their parameters.

    This function returns a dictionary where the keys are the names of linear models (Linear Regression, Lasso, Ridge, Elastic Net) 
    and the values are another dictionary containing the model object and a dictionary of parameters for that model.

    Returns:
    dict: A dictionary of linear models and their parameters.
    """
    return {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'Lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 5000, 10000],
                    'tol': [0.0001, 0.001, 0.01, 0.1],
                    'fit_intercept': [True, False]
                }
            },
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 5000, 10000],
                    'tol': [0.0001, 0.001, 0.01, 0.1],
                    'fit_intercept': [True, False]
                }
            },
            'Elastic Net': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'fit_intercept': [True, False]
                }
            }
        }

def get_model_and_params(results, features, targets):
    """
    Instantiates a model based on the model name and parameters in the results dictionary, fits the model, and returns the model and other parameters.

    This function takes a dictionary of results, which includes the model name, input features, scale, target, and best parameters. 
    It instantiates the appropriate model based on the model name and parameters, fits the model with the scaled features and target, 
    and then returns the model, input features, scale, and target.

    Parameters:
    results (dict): The dictionary of results, which includes the model name, input features, scale, target, and best parameters.
    features (DataFrame): The DataFrame of features.
    targets (Series or DataFrame): The target variable.

    Returns:
    model (object): The fitted model.
    model_input_features (list): The list of input features.
    scale (object): The scaling object.
    target (str): The target variable.
    """
     
    model_name = results['Model']
    model_input_features = results['Features']
    scale = results['scale']
    target = results['target']
    params = results['Best Parameters']

    if model_name == 'Linear Regression':
        model = LinearRegression(**params)
    elif model_name == 'Lasso':
        model = Lasso(**params)
    elif model_name == 'Ridge':
        model = Ridge(**params)
    elif model_name == 'Elastic Net':
        model = ElasticNet(**params)
    else:
        raise ValueError(f"Model type {model_name} not supported")
 
    model.fit(scale.transform(features[model_input_features]), targets[results['target']])

    return model, model_input_features, scale, target

def calculate_correct_sign_percentage(target, predicted_target):
    """
    Calculates the percentage of instances where the target and predicted target have the same sign.

    This function calculates the sign of the target and predicted target, checks if they are equal, 
    counts the number of instances where they are equal, and then calculates the percentage of these instances 
    relative to the total number of instances.

    Parameters:
    target (array-like): The actual target values.
    predicted_target (array-like): The predicted target values.

    Returns:
    correct_sign_percentage (float): The percentage of instances where the target and predicted target have the same sign.
    """

    sign_correct = np.sign(target) == np.sign(predicted_target)
    correct_sign_count = np.sum(sign_correct)
    correct_sign_percentage = (correct_sign_count / len(target)) * 100
    return correct_sign_percentage

def calculate_f_stat_p_val(x_train, target, predicted_target):
    """
    This function calculates the F-statistic and its associated p-value for a regression model. The F-statistic is a measure of how significant the fit of the model is.
    It is calculated by dividing the mean squared error of the model by the mean squared error of the residuals. A higher F-statistic indicates a more significant model.
    The p-value associated with the F-statistic is the probability that the null hypothesis (that the model explains no variance in the data) is true. A lower p-value indicates a more significant model.

    Parameters:
    x_train (pd.DataFrame): The training data.
    target (pd.Series or np.array): The actual target variable for the training data.
    predicted_target (pd.Series or np.array): The predicted target variable for the training data.

    Returns:
    f_statistic (float): The F-statistic for the model.
    p_value (float): The p-value associated with the F-statistic.
    """

    rss = ((target - predicted_target) ** 2).sum()
    tss = ((target - target.mean()) ** 2).sum()
    f_statistic = ((tss - rss) / (len(x_train.columns) - 1)) / (rss / (x_train.shape[0] - len(x_train.columns)))
    p_value = stats.f.sf(f_statistic, len(x_train.columns) - 1, x_train.shape[0] - len(x_train.columns))

    return f_statistic, p_value

def bootstrap_resample(model, x_train, target_train):
    """
    Performs bootstrap resampling on the training data, fits the model, and calculates metrics.

    This function generates bootstrap samples from the training data, fits the model on each sample, 
    calculates the R2 score, mean absolute error (MAE), and percentage of correct signs for each sample, 
    and then calculates the 95% confidence interval for these metrics.

    Parameters:
    model (object): The model to fit.
    x_train (DataFrame): The training data.
    target_train (Series or DataFrame): The target variable for the training data.

    Returns:
    r2_ci_lower (float): The lower bound of the 95% confidence interval for the R2 score.
    r2_ci_upper (float): The upper bound of the 95% confidence interval for the R2 score.
    mae_ci_lower (float): The lower bound of the 95% confidence interval for the MAE.
    mae_ci_upper (float): The upper bound of the 95% confidence interval for the MAE.
    percentage_correct_ci_lower (float): The lower bound of the 95% confidence interval for the percentage of correct signs.
    percentage_correct_ci_upper (float): The upper bound of the 95% confidence interval for the percentage of correct signs.
    bootstrap_r2 (list): The list of R2 scores for each bootstrap sample.
    bootstrap_mae (list): The list of MAEs for each bootstrap sample.
    bootstrap_percentage_correct (list): The list of percentages of correct signs for each bootstrap sample.
    """

    bootstrap_r2 = []
    bootstrap_mae = []
    bootstrap_percentage_correct = []


    n_bootstrap = 1000
    for _ in range(n_bootstrap):
        X_train_resample, target_train_resample = resample(x_train, target_train)
        train_prediction_resample = model.predict(X_train_resample)
        bootstrap_r2.append(r2_score(target_train_resample, train_prediction_resample))
        bootstrap_mae.append(mean_absolute_error(target_train_resample, train_prediction_resample))
        bootstrap_percentage_correct.append(calculate_correct_sign_percentage(target_train_resample, train_prediction_resample))

    r2_ci_lower = round(np.percentile(bootstrap_r2, 2.5), 2)
    r2_ci_upper =  round(np.percentile(bootstrap_r2, 97.5), 2)
    mae_ci_lower =  round(np.percentile(bootstrap_mae, 2.5), 2)
    mae_ci_upper =  round(np.percentile(bootstrap_mae, 97.5), 2)
    percentage_correct_ci_lower =  round(np.percentile(bootstrap_percentage_correct, 2.5), 2)
    percentage_correct_ci_upper =  round(np.percentile(bootstrap_percentage_correct, 97.5), 2)

    return r2_ci_lower, r2_ci_upper, mae_ci_lower, mae_ci_upper, percentage_correct_ci_lower, percentage_correct_ci_upper, bootstrap_r2, bootstrap_mae, bootstrap_percentage_correct

def run_regression(regression_models, x_train, y_train, x_test, y_test, scale = True):
    """
    This function performs model selection and evaluation for a set of regression models. 

    The function takes as input a dictionary of regression models and their associated parameter grids. For each model, 
    it performs a grid search using Leave-One-Out Cross-Validation (LOOCV) and the negative mean absolute error as the scoring metric. 
    LOOCV is a type of cross-validation where each learning set is created by taking all the samples except one, the test set being the sample left out. 
    This is repeated such that each sample in the dataset is used once as the test set. This is particularly useful for small datasets.

    Before each fold in the cross-validation, the features are scaled using StandardScaler. This ensures that the model is not unduly influenced by features 
    with larger scales. The scaler is fitted on the training set of each fold and is used to transform both the training and test sets for that fold.

    The best model for each regression model is selected based on the cross-validation score. The function then fits the best model 
    on the training data and evaluates its performance on both the training and test data. The performance metrics calculated are the R2 score 
    and the mean absolute error (MAE). The R2 score is a statistical measure that represents the proportion of the variance for a dependent variable 
    that's explained by an independent variable or variables in a regression model. The MAE measures the average magnitude of the errors in a set of predictions, 
    without considering their direction.

    In addition to these, the function also calculates the mean and standard deviation of the MAE scores from the LOOCV. It also computes the 95% confidence 
    interval for the mean MAE. The 95% confidence interval is a range of values that you can be 95% certain contains the true mean of the population.

    Finally, the function returns a DataFrame that contains the model name, best parameters, best cross-validation score, training and test R2 scores, 
    training and test MAEs, mean and standard deviation of the LOOCV MAE scores, 95% confidence interval for the mean MAE, and the features used for each model.

    Parameters:
    regression_models (dict): A dictionary where the keys are model names and the values are dictionaries containing the model 
                            instance and the parameter grid for GridSearchCV.
    x_train (pd.DataFrame): The training data.
    y_train (pd.Series or np.array): The target variable for the training data.
    x_test (pd.DataFrame): The test data.
    y_test (pd.Series or np.array): The target variable for the test data.

    Returns:
    results_df (pd.DataFrame): A DataFrame containing the model name, best parameters, best cross-validation score, training and test R2 scores, 
                            training and test MAEs, mean and standard deviation of the LOOCV MAE scores, 95% confidence interval for the mean MAE, 
                            and the features used for each model.
    """
    results = []
    r2_dict = {}

    for model_name, model_info in regression_models.items():
        if scale:
            scaler = StandardScaler()
        else:
            scaler = FunctionTransformer(func = lambda x: x)
        pipeline = Pipeline(steps=[('scaler', scaler), ('model', model_info['model'])])
        loocv = LeaveOneOut()
        params = {'model__' + key: value for key, value in model_info['params'].items()}
        grid_search = GridSearchCV(pipeline, params, cv=loocv, scoring='neg_mean_absolute_error', return_train_score=False)
        grid_search.fit(x_train, y_train)

        scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=loocv, scoring='neg_mean_absolute_error')
        cv_predictions = cross_val_predict(grid_search.best_estimator_, x_train, y_train, cv=loocv)
        r2_cv = r2_score(y_train, cv_predictions)
        
        mae_scores = -scores

        mean_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)

        ci_lower = mean_mae - 1.96 * std_mae / np.sqrt(len(mae_scores))
        ci_upper = mean_mae + 1.96 * std_mae / np.sqrt(len(mae_scores))

        train_prediction = grid_search.predict(x_train)
        test_prediction = grid_search.predict(x_test)

        r2_train = round(r2_score(y_train, train_prediction) ,2)
        r2_test = round(r2_score(y_test, test_prediction) ,2)
        
        mae_train = round(mean_absolute_error(y_train, train_prediction),2)
        mae_test = round(mean_absolute_error(y_test, test_prediction),2)

        percentage_train = round(calculate_correct_sign_percentage(y_train, train_prediction),2)
        percentage_test = round(calculate_correct_sign_percentage(y_test, test_prediction),2)

        f_statistic, p_value = calculate_f_stat_p_val(x_train, y_train, grid_search.predict(x_train))

        r2_ci_lower, r2_ci_upper, mae_ci_lower, mae_ci_upper, percentage_correct_ci_lower, percentage_correct_ci_upper, r2_scores, maes, correct_signs = bootstrap_resample(grid_search, x_train, y_train)

        r2_dict[model_name] = r2_scores
     
        results.append({
        'Model': model_name,
        'Best Parameters': {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()},
        'Best CV Score': grid_search.best_score_,
        'Train R2': r2_train,
        'Train R2 95% CI': (r2_ci_lower, r2_ci_upper),
        'Test R2': r2_test,
        'Train MAE': mae_train,
        'Train MAE 95% CI': (mae_ci_lower, mae_ci_upper),
        'Test MAE': mae_test,       
        'F-statistic': f_statistic,
        'F-statistic p-value': p_value,
        'Correct sign Train': percentage_train,
        'Correct sign Test': (percentage_test, percentage_correct_ci_lower, percentage_correct_ci_upper),
        'Correct sign Test': percentage_test,
        'Features': x_train.columns.tolist(),
        'LOOCV Mean MAE': mean_mae,
        'LOOCV Std MAE': std_mae,
        'LOOCV 95% CI Lower': ci_lower,
        'LOOCV 95% CI Upper': ci_upper,
        'LOOCV R2': r2_cv
        })

    results_df = pd.DataFrame(results)
    
    return results_df, r2_dict, grid_search.best_estimator_.named_steps['scaler']

def train(linear_models, features_train, features_test, target_train, target_test, targets):
    reg_reults_dict = {}

    for target in targets:
        X_train = features_train.copy()
        X_test = features_test.copy()
        if target in ['ChronAge','PhenotypicAge', 'PCPhenoAge']:
            logging.info('raw age, must drop chron age')
            # remove chronage from features train
            X_train = X_train.drop("ChronAge", axis = 1)
            X_test = X_test.drop("ChronAge", axis = 1)
        else:
            pass  
        t = time.time()
        logging.info(f'working on {target}')
        behav = ['weekly light active hours', 'weekly moderate active hours', 'weekly vigorous active hours', 'daily sedentary hours']
        active = ['weekly light active hours', 'weekly moderate active hours', 'weekly vigorous active hours']
        phys  = ['VO2max', 'normalized crest time','SEVR', 'SAR','LASI', 'BPI', 'ED', 'RHR']
        circadian_pca = ['sleep PCA1', 'sleep PCA2']

        if target == 'PCPhenoAgeResid':
            selected_features_dict = {
                # "all": X_train.columns.tolist(),
                "physiological": phys,
                # "physiological_behavioural": phys + behav,
                # "physiological_activity": phys + active,
                # "physiological_circadian": phys + circadian_pca,
                # "circadian" : circadian_pca,
                # "behavioural": behav,
                # "behavioural_circadian" : behav + circadian_pca  
                }
        elif target == 'PCPhenoAge':
            selected_features_dict = {
                "all": X_train.columns.tolist(),
                # "physiological": phys,
                # "physiological_behavioural": phys + behav,
                # "physiological_activity": phys + active,
                # "physiological_circadian": phys + circadian_pca,
                # "circadian" : circadian_pca,
                # "behavioural": behav,
                # "behavioural_circadian" : behav + circadian_pca  
            }
        else:
            selected_features_dict = {
                                    "all": X_train.columns.tolist(),
                                    "phys": phys,
                                    'behav': behav,
                                    "circ_pca": circadian_pca
                                    }
        logging.info(f"selected_features dictionary: {selected_features_dict}")

        overall_results =[]
        
        for name, selected_features in selected_features_dict.items():
            results, r2_dict, scale = run_regression(linear_models, X_train[selected_features], target_train[target], X_test[selected_features], target_test[target], scale=True)
            results.sort_values(by=['Test R2'], ascending=False)
            results['feature_selection'] = name
            results['target'] = target
            results['scale'] = scale
            overall_results.append(results)
            reg_df = pd.concat(overall_results)
            reg_df = reg_df.sort_values(by=['Test R2', 'Train R2'], ascending=[False, False])
            reg_reults_dict[target] = reg_df

            logging.info(f'duration:, {time.time()-t}')
            
    return reg_reults_dict

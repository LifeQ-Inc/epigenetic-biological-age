import logging
import argparse
import pickle
from src.interpretation import *
from src.utils import *
from src.pca import *
from src.training import *
from src.plotting import *

logging.basicConfig(level=logging.INFO)

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if not os.path.exists('models'):
        logging.info('making model directory')
        os.makedirs('models')
    if not os.path.exists('plots'):
        logging.info('making plots directory')
        os.makedirs('plots')

    logging.info("Reading data")
    df, features, age_df, sleep_features = read_data()

    if age_df.empty:
        logging.error('Process epigenetic data first')
        raise Exception("Process epigenetic data first")

    if 'sleep_PCA1' in df.columns:
        features['sleep PCA1'] = df['sleep_PCA1']
        features['sleep PCA2'] = df['sleep_PCA2']
    else:
        pca_df, loadings_df, important_features_pca1, important_features_pca2 = run_pca_process(sleep_features, 2, drop=True)
        features['sleep PCA1'] = pca_df['PCA1']
        features['sleep PCA2'] = pca_df['PCA2']

    if args.train:
       
        features_train, features_test, target_train, target_test = prepare_data(features, age_df, test_set_size=0.2, random_seed=0, stratification=True)
        plot_target_distribution(target_train, target_test)

        linear_models = get_linear_models()
        targets = ['PCPhenoAgeResid', 'PCPhenoAge']
        results = train(linear_models, features_train, features_test, target_train, target_test, targets)

        pheno_accel_results = results['PCPhenoAgeResid'][results['PCPhenoAgeResid']['feature_selection'] == 'physiological']
        pheno_results = results['PCPhenoAge']
        pheno_accel_results.to_csv('data/PCPhenoAgeAccel_model_performance.csv')
        pheno_results.to_csv('data/PCPhenoAge_model_performance.csv')

        model, model_input_features, scale, target = get_model_and_params(pheno_accel_results.iloc[0], features_train, target_train)

        model_info = {
            'model': model,
            'model_input_features': model_input_features,
            'scale': scale,
            'target': target
        }

        with open(f'models/{target}_model.pkl', 'wb') as f:
            pickle.dump(model_info, f)

        train_indices = set(features_train.index)
        test_indices = set(features_test.index)
  
        age_df['Dataset'] = age_df.index.map(lambda x: 'Train' if x in train_indices else ('Test' if x in test_indices else 'Unknown'))
        features['Dataset'] = features.index.map(lambda x: 'Train' if x in train_indices else ('Test' if x in test_indices else 'Unknown'))
        features_scaled = scale.transform(features[model_input_features])

        age_df[f'predicted_{target}'] = model.predict(features_scaled)
        plot_model_predictions_scatter_and_distribution(age_df, target)

    else:
        model_info = pickle.load(open('models/PCPhenoAgeResid_model.pkl', 'rb'))
        model, model_input_features, scale, target = extract_model_info(model_info)

        features_scaled = scale.transform(features[model_input_features])
        age_df[f'predicted_{target}'] = model.predict(features_scaled)

    importance = model.coef_
    feature_importance = pd.DataFrame(importance, index=model_input_features, columns=['importance'])

    plot_shap_violin_plot(importance, model_input_features, features_scaled, features, model)
    individual_feature_contribution_plot(features_scaled, model_input_features, feature_importance, age_df, target, participant = 7)

    age_df.to_csv('data/predictions.csv')

if __name__ == '__main__':
    main()
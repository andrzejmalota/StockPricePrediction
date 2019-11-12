import xgboost as xgb
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.io import load, save
from src.visualization.visualize import *


def get_X_y(data):
    X = data[:, 1:]
    y = data[:, 1]
    return X, y


def get_sliding_windows(X, y, lookback, delay, min_index, max_index):
    f = False
    for i in range(min_index, max_index):
        if not f:
            samples = X[i:i + lookback].flatten()
            samples = samples.reshape((1, samples.shape[0]))
            targets = y[i + lookback:i + lookback + delay]
            f = True
        else:
            temp = X[i:i + lookback].flatten()
            samples = np.r_[samples, temp.reshape((1, temp.shape[0]))]
            targets = np.r_[targets, y[i + lookback:i + lookback + delay]]
    return samples, targets


def get_random_forrest_model(samples, targets):
    regressor = RandomForestRegressor(criterion='mae', n_estimators=50, verbose=True, n_jobs=-1)
    model = regressor.fit(samples, targets)
    return regressor


def get_xgb_model(samples, targets, samples_eval, targets_eval):
    regressor = xgb.XGBRegressor()
    model = regressor.fit(samples, targets, early_stopping_rounds=3, eval_metric="mae",
                          eval_set=[(samples, targets), (samples_eval, targets_eval)], verbose=True)
    return regressor


def split_data(data, lookback):
    values = data.to_numpy()
    X, y = get_X_y(values)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(X)
    X = imp_mean.transform(X)

    cut = 1
    if cut:
        # cutoff first 700 data points
        X = X[700:]
        y = y[700:]
    train_size = int(0.8 * X.shape[0])
    samples, targets = get_sliding_windows(X, y, lookback, 1, 0, train_size)
    samples_eval, targets_eval = get_sliding_windows(X, y, lookback, 1, train_size, X.shape[0] - lookback)
    return (samples, targets), (samples_eval, targets_eval)


def build_model(data, lookback):
    (samples, targets), (samples_eval, targets_eval) = split_data(data, lookback)
    # Select model RandomForrest or XGBooster
    # model = get_random_forrest_model(samples, targets)
    model = get_xgb_model(samples, targets, samples_eval, targets_eval)
    predictions = model.predict(samples_eval)
    mse = mean_squared_error(targets_eval, predictions)
    mae = mean_absolute_error(targets_eval, predictions)
    print("MSE: ", mse)
    print("MAE: ", mae)
    # plot_targets_vs_predictions(targets_eval, predictions)
    return model


def create_feature_labels(features, lookback):
    feature_labels = []
    features = features.columns[1:]
    for i in range(lookback, 0, -1):
        feature_labels += [feature + '_' + str(i) for feature in features]
    print('n_features: ', len(feature_labels))
    feature_labels = np.asarray(feature_labels)
    return feature_labels


def calculate_feature_importances(features, lookback, model):
    feature_labels = create_feature_labels(features, lookback)
    importances = model.feature_importances_.astype('float32')
    indices = np.argsort(-importances)
    values = np.c_[feature_labels[indices].reshape(feature_labels.shape[0], 1),
                   importances[indices].reshape(feature_labels.shape[0], 1)]
    feature_importances = pd.DataFrame(data=values, columns=['feature_labels', 'feature_importance'])
    return feature_importances


def threshold_features(feature_importances, lookback):
    threshold = 0.0002
    feature_importances['feature_importance'] = feature_importances['feature_importance'].astype('float32')
    filtered_features = feature_importances[feature_importances['feature_importance'] > threshold]['feature_labels']
    selected_features = []
    for feature in filtered_features:
        selected_features.append(feature.replace('_'+feature.split('_')[-1], ''))
    return list(set(selected_features)), filtered_features


def select_features():
    features = load('../../data/interim/features.pickle')
    lookback = 10
    model = build_model(features, lookback)
    feature_importances = calculate_feature_importances(features, lookback, model)
    # plot_feature_importance(feature_importances)
    selected_features, selected_features_with_lookback = threshold_features(feature_importances, lookback)
    print('Selected features with trees: ', selected_features)
    not_selected_features = list(set(features.columns.to_list()).difference(set(selected_features)))
    print('Not selected features: ', not_selected_features)
    save(selected_features, '../../data/interim/selected_features_labels_trees.pickle')
    save(features[selected_features], '../../data/processed/selected_features_trees.pickle')
    save(selected_features_with_lookback, '../../data/interim/selected_features_labels_with_lookback_trees.pickle')


if __name__ == '__main__':
    select_features()

import xgboost as xgb
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.io import load, save


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
    plot_targets_vs_predictions(targets_eval, predictions)
    return model


def create_feature_labels(features, lookback):
    feature_labels = []
    features = features.columns[1:]
    for i in range(lookback, 0, -1):
        feature_labels += [feature + '_' + str(i) for feature in features]
    print('n_features: ', len(feature_labels))
    feature_labels = np.asarray(feature_labels)
    return feature_labels


def calculate_feature_importances(features, lookback):
    model = build_model(features, lookback)
    feature_labels = create_feature_labels(features, lookback)
    feature_df = pd.DataFrame(data=fea)
    importances = model.feature_importances_.astype('float32')
    indices = np.argsort(-importances)
    values = np.c_[feature_labels[indices].reshape(feature_labels.shape[0], 1),
                   importances[indices].reshape(feature_labels.shape[0], 1)]
    feature_importances = pd.DataFrame(data=values, columns=['feature_labels', 'feature_importances'])
    return feature_importances, features_df


def plot_targets_vs_predictions(targets, predictions):
    targets = targets.tolist()
    predictions = [round(y, 2) for y in predictions]
    fig = plt.figure(figsize=(20, 8))
    plt.plot(targets, label='targets')
    plt.plot(predictions, label='predictions')
    plt.title('Targets vs predicitons')
    plt.legend()
    plt.show()


def plot_validation_vs_training(model):
    eval_result = model.evals_result()
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    plt.scatter(x=training_rounds, y=eval_result['validation_0']['rmse'], label='Training Error')
    plt.scatter(x=training_rounds, y=eval_result['validation_1']['rmse'], label='Validation Error')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Training Vs Validation Error')
    plt.legend()
    plt.show()


def plot_feature_importance(feature_importances):
    rc('xtick', labelsize=6)
    rc('ytick', labelsize=6)
    fig = plt.figure(figsize=(10, 100))
    plt.xticks(rotation='vertical')
    plt.barh(range(100), feature_importances.iloc[:100, 1])
    plt.yticks(range(100), feature_importances.iloc[:100, 0])
    plt.title('Feature importance')
    plt.show()


if __name__ == '__main__':
    features = load('../../data/interim/features.pickle')
    lookback = 10
    feature_importances = calculate_feature_importances(features, lookback)
    plot_feature_importance(feature_importances)
    save(feature_importances, '../../data/interim/feature_importances_by_trees.pickle')


# def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             i += len(rows)
#         samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + delay][1]
#         yield samples, targets

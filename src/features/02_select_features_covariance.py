from src.utils.io import load, save
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def select_features():
    features = load('../../data/interim/features.pickle')
    features.drop('Date', inplace=True, axis=1)
    features_diff = features.diff()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(features_diff)
    features_diff = imp_mean.transform(features_diff)
    scaler = StandardScaler()
    features_diff = scaler.fit_transform(features_diff)
    features_diff = pd.DataFrame(data=features_diff, columns=features.columns)
    features_cov = features_diff.cov()
    close_price_cov = pd.DataFrame(features_cov['Close'].apply(lambda x: abs(x)).sort_values(ascending=False))
    selected_features = close_price_cov[close_price_cov['Close'] > 0.1].index.to_list()
    print('Selected features with covariance: ', selected_features)
    not_selected_features = list(set(features.columns.to_list()).difference(set(selected_features)))
    print('Not selected features: ', not_selected_features)
    save(selected_features, '../../data/interim/selected_features_labels_cov.pickle')
    save(features[selected_features], '../../data/processed/selected_features_cov.pickle')
    print()


if __name__ == '__main__':
    select_features()

from src.utils.io import load, save
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
import pandas as pd


def select_features():
    features = load('../../data/interim/features_amazon.pickle')
    date = features['Date']
    features.drop('Date', inplace=True, axis=1)

    # calculate difference between rows
    features_diff = features.pct_change().replace([np.inf, -np.inf], np.nan)

    # impute NaN values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(features_diff)
    features_diff = imp_mean.transform(features_diff)

    # features_diff.dropna()
    features_diff = pd.DataFrame(data=features_diff, columns=features.columns)

    corr_matrix = features_diff.corr().abs()

    # calculate covariance matrix
    close_price_cov = pd.DataFrame(corr_matrix['Close'].sort_values(ascending=False))

    # select features
    cov_threshold = 0.001
    selected_features = close_price_cov[close_price_cov['Close'] > cov_threshold].index.to_list()
    # print('Selected features with covariance: ', selected_features)
    not_selected_features = list(set(features.columns.to_list()).difference(set(selected_features)))
    print('Not selected features: ', not_selected_features)

    save(selected_features, '../../data/interim/selected_features_labels_cov.pickle')
    features = features[selected_features]

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    for f in not_selected_features:
        if f in to_drop:
            to_drop.remove(f)

    features.drop(to_drop, axis=1, inplace=True)
    print(to_drop)

    # normalize
    # for col in features_diff.columns:
    #     features_diff[col] = preprocessing.scale(features_diff[col].values)
    #

    #
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # features = pd.DataFrame(data=imp_mean.fit_transform(features), columns=features.columns)

    # ftrs_pct = features.pct_change()
    # ftrs_pct = ftrs_pct.replace([np.inf, -np.inf], np.nan)
    # ftrs_pct = pd.DataFrame(data=imp_mean.fit_transform(ftrs_pct), columns=ftrs_pct.columns)
    # ftrs_pct.astype(np.float64)
    # ftrs_pct['Date'] = date
    features['Date'] = date
    features.set_index('Date', inplace=True)
    save(features, '../../data/processed/features_amazon_corr.pickle')
    print()


if __name__ == '__main__':
    select_features()

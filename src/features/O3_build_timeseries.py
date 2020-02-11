import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.utils.io import load, save


def drop_not_relevant(features, targets, days):
    features = features.iloc[days:]
    targets = targets.iloc[days:]
    return features, targets


def impute(features):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    features_imputed = pd.DataFrame(imp_mean.fit_transform(features), columns=features.columns)
    return features_imputed


def build_timeseries_windows(dataset, target, start_index, end_index, lookback,
                             target_size, step, single_step=True):
    data = []
    labels = []

    start_index = start_index + lookback
    if end_index is None:
        end_index = min(len(dataset), len(target)) - target_size

    for i in range(start_index, end_index):
        indices = range(i-lookback, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def scale(data, train_size):
    data_mean = data[:-train_size].mean(axis=0)
    data_std = data[:-train_size].std(axis=0)
    data = (data-data_mean)/data_std
    return data


def train_val_test_split(x, y, test_size, val_size):
    test_x = x[-test_size:]
    test_y = y[-test_size:]
    val_x = x[-(test_size+val_size):-test_size]
    val_y = y[-(test_size+val_size):-test_size]
    train_x = x[:-(test_size+val_size)]
    train_y = y[:-(test_size+val_size)]

    print(train_x.shape)
    print(val_x.shape)
    print(test_x.shape)

    return train_x, train_y, val_x, val_y, test_x, test_y


def check_if_data_valid(dataset, lookback):
    train_x, train_y, val_x, val_y, test_x, test_y = dataset
    for data_x, data_y in zip([train_x, val_x, test_x], [train_y, val_y, test_y]):
        data_x_bin = (data_x >= 0).astype(np.int32)
        for i in range(0, data_y.shape[0]-lookback):
            assert data_x_bin[i+lookback, 0, 0] == data_y[i]


def build_data(features, targets, lookback=1, test_size=250, val_size=250, encode_binary=True,
               scaled=True, check_data=True, pct_change=True):
    name = 'data_lookback_' + str(lookback)
    future_target = 1
    step = 1
    if pct_change:
        targets = pd.DataFrame(targets['Close'].pct_change(), columns=['Close'])
    else:
        targets = pd.DataFrame(targets['Close'], columns=['Close'])

    if encode_binary:
        bin_labels = (targets['Close'] >= 0).astype(np.int32).values
    else:
        bin_labels = targets.values
        name += '_notbinary'

    cols = features.columns.to_list()
    cols.remove('Close')
    features = features[['Close'] + cols]

    features = impute(features)

    # Modify features by calculating 1 day percentage change
    if pct_change:
        features = features.pct_change()
        features = features.replace([np.inf, -np.inf], np.nan)
        features = impute(features)

    # Scale feature with normal scaler
    if scaled:
        data = scale(features, test_size + val_size).values
    else:
        data = features.values
        name += '_notscaled'

    # Build timeseries windows with width = lookback
    x, y = build_timeseries_windows(data[1:], bin_labels[1:], 0, None, lookback, future_target, step)

    dataset = train_val_test_split(x, y, test_size, val_size)

    if check_data:
        if encode_binary:
            x_, y_ = build_timeseries_windows(features.values[1:], bin_labels[1:], 0, None, lookback, future_target, step)
            dataset_ = train_val_test_split(x_, y_, test_size, val_size)
            check_if_data_valid(dataset_, lookback)

    return dataset, name


def build_default_data():
    features = load('../../data/processed/features_amazon_corr.pickle')
    targets = load('../../data/processed/targets_amazon.pickle')
    not_relevant_days = 1660
    features, targets = drop_not_relevant(features, targets, not_relevant_days)
    test_size = int(0.05 * features.shape[0])
    val_size = int(0.15 * features.shape[0])

    for encode_binary in [True, False]:
        for scaled in [True, False]:
            for lookback in [1, 60]:
                data, name = build_data(features,
                                        targets,
                                        lookback=lookback,
                                        scaled=scaled,
                                        encode_binary=encode_binary,
                                        test_size=test_size,
                                        val_size=val_size,
                                        pct_change=True)
                save(data, '../../data/timeseries/' + name + '_amazon.pickle')


def build_data_trading_plot():
    features = load('../../data/processed/features_amazon_corr.pickle')
    targets = load('../../data/processed/targets_amazon.pickle')
    not_relevant_days = 1660
    features, targets = drop_not_relevant(features, targets, not_relevant_days)
    test_size = int(0.05 * features.shape[0])
    val_size = int(0.15 * features.shape[0])

    data, name = build_data(features,
                            targets,
                            lookback=1,
                            scaled=False,
                            encode_binary=False,
                            test_size=test_size,
                            val_size=val_size,
                            pct_change=False)
    save(data, '../../data/timeseries/' + name + '_trading_vis_amazon.pickle')


if __name__ == '__main__':
    build_default_data()
    build_data_trading_plot()

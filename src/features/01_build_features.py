# 1. Technical indicators:
#     *   7,21 moving average
#     *   exponential moving average
#     *   momentum
#     *   Bollinger bands
#     *   MACD
# 2. ARIMA
# 3. Correlated assets:
#     -companies similar to Tesla
#     -Daily volatility index (VIX)
# 4.Fourier Transform for trend analysis
# 5. Close and open returns  (10 day, market-residualized return)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ta.volume import acc_dist_index, on_balance_volume
from ta.momentum import rsi
import datapackage
from src.utils.io import load, save


def get_technical_indicators(dataset, key='Close'):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset[key].rolling(window=7).mean()
    dataset['ma21'] = dataset[key].rolling(window=21).mean()

    # Create MACD -> Moving Average Convergence/Divergence (trend and momentum indicator)
    dataset['26ema'] = dataset[key].ewm(span=26).mean()
    dataset['12ema'] = dataset[key].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Create Bollinger Bands
    # Bollinger Bands are volatility bands placed above and below a moving average. Volatility is based on the standard deviation, which changes as volatility 
    # increases and decreases. The bands automatically widen when volatility increases and narrow when volatility decreases.
    dataset['20std'] = dataset[key].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20std'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20std'] * 2)

    # Create Exponential moving average
    dataset['ema'] = dataset[key].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum_1'] = dataset[key].diff(1)
    dataset['momentum_10'] = dataset[key].diff(10)

    # close price raw return 1, 10 days horizon
    # dataset['returnsClosePrevRaw1'] = dataset['Close'].pct_change(1)
    # dataset['returnsClosePrevRaw10'] = dataset['Close'].pct_change(10)

    # open price raw return 1, 10 days horizon
    # dataset['returnsOpenPrevRaw1'] = dataset['Open'].pct_change(1)
    # dataset['returnsOpenPrevRaw10'] = dataset['Open'].pct_change(10)

    # Create AROON - identifing when trends are likely to change direction, n=20
    dataset['aroon_up_20'] = dataset[key].rolling(20, min_periods=0).apply(lambda x: float(np.argmax(x) + 1) / 20 * 100,
                                                                           raw=True)
    dataset['aroon_down_20'] = dataset[key].rolling(20, min_periods=0).apply(
        lambda x: float(np.argmin(x) + 1) / 20 * 100, raw=True)

    # Create CCI, Commodity Channel Index
    pp = (dataset['High'] + dataset['Low'] + dataset[key]) / 3.0
    dataset['cci'] = (pp - pp.rolling(20, min_periods=0).mean()) / (0.015 * pp.rolling(20, min_periods=0).std())

    # Create STOCH - Stochastic Oscillator
    smin = dataset['Low'].rolling(14, min_periods=0).min()
    smax = dataset['High'].rolling(14, min_periods=0).max()
    dataset['stoch'] = 100 * (dataset[key] - smin) / (smax - smin)

    # Create RSI - Relative Strength Index
    dataset['rsi'] = rsi(dataset[key])

    # Create ADI - Accumulation/Distribution Index
    dataset['adi'] = acc_dist_index(dataset['High'], dataset['Low'], dataset[key], dataset['Volume'])

    # Create OBV - On-balance volume
    dataset['obv'] = on_balance_volume(dataset[key], dataset['Volume'])

    return dataset


def get_corr_assets(dataset):
    data_url = 'https://datahub.io/core/finance-vix/datapackage.json'

    # to load Data Package into storage
    package = datapackage.Package(data_url)

    # to load only tabular data
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            data = pd.read_csv(resource.descriptor['path'])
            break
    data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
    start = data[data['Date'] == dataset['Date'][0]].index[0]

    # Create VIX Close and Open
    dataset['vixClose'] = data['VIX Close'][start:].reset_index().drop('index', axis=1)
    dataset['vixOpen'] = data['VIX Open'][start:].reset_index().drop('index', axis=1)
    return dataset


def get_fourier_transforms(dataset):
    """ Trend extraction and filtering out noise """
    data_FT = dataset[['Date', 'Close']]
    close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
    for num in [3, 6, 9, 50, 100, 200, 500]:
        ifft = np.copy(close_fft)
        ifft[num:-num] = 0
        dataset[f'fft{num}'] = np.real((np.fft.ifft(ifft)))
    # for num in [3, 6, 9]:
    #     ifft = np.copy(close_fft)
    #     ifft[num:-num] = 0
    #     dataset[f'fft{num}'] = np.real((np.fft.ifft(ifft)))

    # PLOTTING
    fft_df = pd.DataFrame({'fft': close_fft})
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100, 500]:
        fft_list_m10 = np.copy(fft_list); fft_list_m10[num_:-num_]=0
        plt.plot(np.fft.ifft(fft_list_m10), label='Transformata Fouriera z {} komponenetami'.format(num_))
    plt.plot(data_FT['Close'],  label='Cena')
    plt.xlabel('Data')
    plt.ylabel('USD')
    plt.title('Cena zamknięcia oraz transformata Fouriera')
    plt.legend()
    plt.show()
    return dataset


def get_automotive_industry_close_prices(dataset, raw_stock_data):
    tesla = raw_stock_data['tesla']
    for company in ['bmw', 'daimler', 'porshe']:
        other = raw_stock_data[company]
        start = other[other['Date'] == tesla['Date'][0]].index[0]
        dataset[company + '_close'] = other['Close'][start:].reset_index().drop('index', axis=1)
    return dataset


def build_features():
    data = load('../../data/raw/stock_data.pickle')
    # features = data['tesla']
    features = data['amazon']
    features = get_technical_indicators(features)
    # features = get_corr_assets(features)
    features = get_fourier_transforms(features)
    # features = get_automotive_industry_close_prices(features, data)
    # save(features, '../../data/interim/features.pickle')
    save(features, '../../data/interim/features_amazon.pickle')


if __name__ == '__main__':
    build_features()

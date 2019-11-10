import matplotlib.pyplot as plt
import numpy as np
from src.utils.io import load


def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(24, 10))
    end_idx = dataset.shape[0]
    start_idx = end_idx - last_days
    dataset = dataset.iloc[-last_days:, :]

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    plt.plot(dataset['Close'], label='Closing Price', color='k')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='b')
    plt.plot(dataset['lower_band'], label='Lower Band', color='b')
    plt.plot(dataset['26ema'], label='EMA 26', color='y')
    plt.fill_between(dataset.index, dataset['lower_band'], dataset['upper_band'], alpha=0.1)
    plt.title('Technical indicators for Tesla - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(20, start_idx, end_idx, colors='g', linestyles='--')
    plt.hlines(-20, start_idx, end_idx, colors='g', linestyles='--')
    plt.plot(np.log(dataset['momentum']), label='Log Momentum', color='b', linestyle='-')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    features = load('../../data/interim/features.pickle')
    plot_technical_indicators(features, 1000)

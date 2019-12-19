import matplotlib.pyplot as plt
from matplotlib import rc
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
    plt.plot(dataset['Close'], label='Cena zamknięcia', color='k')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Górna wstęga Bollinger', color='b')
    plt.plot(dataset['lower_band'], label='Dolna wstęga Bollinger', color='b')
    plt.plot(dataset['26ema'], label='EMA 26', color='y')
    plt.fill_between(dataset.index, dataset['lower_band'], dataset['upper_band'], alpha=0.1)
    # plt.title('Techniczne in {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    plt.hlines(20, start_idx, end_idx, colors='g', linestyles='--')
    plt.hlines(-20, start_idx, end_idx, colors='g', linestyles='--')
    plt.plot(np.log(dataset['momentum_10']), label='Log Momentum', color='b', linestyle='-')

    plt.legend()
    plt.show()


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
    fig = plt.figure(figsize=(10, 10))
    plt.xticks(rotation='vertical')
    plt.barh(range(100), feature_importances.iloc[:100, 1])
    plt.yticks(range(100), feature_importances.iloc[:100, 0])
    plt.title('Feature importance')
    plt.show()


if __name__ == '__main__':
    features = load('../../data/interim/features.pickle')
    plot_technical_indicators(features, 500)

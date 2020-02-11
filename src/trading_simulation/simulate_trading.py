from src.trading_simulation.simulation import Simulation
from src.trading_simulation.strategy import SimpleStrategy, BuyAndHold
from src.utils.io import load, save
import numpy as np
import pandas as pd


if __name__ == '__main__':
    init_investment = 5000
    # targets = load('../../data/processed/targets.pickle')
    targets = load('../../data/processed/targets_amazon.pickle')
    date = pd.DataFrame(targets['Date'])

    # Load model predictions
    dense = load('../../data/predictions/dense.pickle')
    gru = load('../../data/predictions/gru.pickle')
    # gru = load('../../data/predictions/gru_amazon.pickle')
    pseudo_random = load('../../data/predictions/pseudo_random.pickle')
    # pseudo_random = load('../../data/predictions/pseudo_random_amazon.pickle')
    lstm = load('../../data/predictions/lstm.pickle')
    conv_lstm = load('../../data/predictions/conv_lstm.pickle')
    models = [pseudo_random, dense, gru, lstm, conv_lstm]
    model_names = ['pseudo_random', 'dense', 'gru', 'lstm', 'conv_lstm']

    # load stock return for test set
    stock_returns = load('../../data/timeseries/data_lookback_1_notbinary_notscaled.pickle')[5]
    # stock_returns = load('../../data/timeseries/data_lookback_1_notbinary_notscaled_amazon.pickle')[5]
    stock_returns = stock_returns.reshape(1, stock_returns.shape[0]).tolist()[0]

    # load stock prices for test set
    stock_prices = load('../../data/timeseries/data_lookback_1_notbinary_notscaled_trading_vis.pickle')[5]
    # stock_prices = load('../../data/timeseries/data_lookback_1_notbinary_notscaled_trading_vis_amazon.pickle')[5]

    trading_return = []
    trading_profit = []

    # Run simulation for pseudo-random model's predictions
    print(model_names[0])
    for model_predictions in models[0]:
        trading_simulation = Simulation(init_investment, stock_returns, SimpleStrategy(), list(model_predictions))
        trading_simulation.start()
        trading_performance = trading_simulation.get_investment_performance()
        trading_profit.append(trading_performance['profit'])
        trading_return.append(trading_performance['return'])

    # Calculate mean return and profit over all predictions
    trading_performance['return'] = np.mean(trading_return)
    trading_performance['profit'] = np.mean(trading_profit)
    print(round(100*trading_performance['return'], 2), '%, ', round(trading_performance['profit'], 2), '$')
    print()
    trading_simulation.plot_trading_history(stock_prices, date)

    # Run simulation for rest of the models predictions
    for model_name, model_predictions in zip(model_names[1:], models[1:]):
        print(model_name)
        trading_simulation = Simulation(init_investment, stock_returns, SimpleStrategy(), list(model_predictions))
        trading_simulation.start()
        trading_performance = trading_simulation.get_investment_performance()
        print(round(100*trading_performance['return'], 2), '%, ', round(trading_performance['profit'], 2), '$')
        trading_simulation.plot_trading_history(stock_prices, date)
        print()

    # Run simulation for BuyAndHold strategy
    strategy = BuyAndHold(len(stock_returns))
    print('Buy&Hold')
    trading_simulation = Simulation(init_investment, stock_returns, strategy)
    trading_simulation.start()
    trading_performance = trading_simulation.get_investment_performance()
    print(round(100*trading_performance['return'], 2), '%, ', round(trading_performance['profit'], 2), '$')
    trading_simulation.plot_trading_history(stock_prices, date)


    print(1)

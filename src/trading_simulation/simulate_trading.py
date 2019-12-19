from src.trading_simulation.simulation import Simulation
from src.trading_simulation.strategy import SimpleStrategy
from src.utils.io import load, save

if __name__ == '__main__':
    init_investment = 10000
    dense = load('../../data/predictions/dense.pickle')
    gru = load('../../data/predictions/gru.pickle')
    pseudo_random = load('../../data/predictions/pseudo_random.pickle')
    lstm = load('../../data/predictions/lstm.pickle')
    conv_lstm = load('../../data/predictions/conv_lstm.pickle')

    stock_returns = load('../../data/timeseries/data_lookback_1_notbinary_notscaled.pickle')[5]
    stock_returns = stock_returns.reshape(1, stock_returns.shape[0]).tolist()[0]

    stock_prices = load('../../data/timeseries/data_lookback_1_notbinary_notscaled_trading_vis.pickle')[5]

    strategy = SimpleStrategy()

    for predicted_movements in [pseudo_random, dense, gru, lstm, conv_lstm]:
        trading_simulation = Simulation(init_investment, list(predicted_movements), stock_returns, strategy)
        trading_simulation.start()
        trading_performance = trading_simulation.get_investment_performance()
        print(trading_performance)
        # trading_simulation.plot_trading_history(stock_prices)
        print()

    print(1)

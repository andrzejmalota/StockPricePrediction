import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Simulation:
    def __init__(self, init_investment, stock_returns, strategy, predicted_movements=None):
        self.init_investment = init_investment
        self.predicted_movements = predicted_movements
        self.stock_returns = stock_returns
        self.strategy = strategy
        self.action_history = []
        self.account_history = [init_investment]
        self.__actual_investment = 0
        self.step = 0
        self.return_on_investment = 0
        self.profit_on_investment = 0

    def start(self):
        for self.step in range(len(self.stock_returns)):
            if self.predicted_movements is not None:
                action = self.strategy.decide(self.predicted_movements[self.step])
            else:
                action = self.strategy.decide(self.step)
            self.__make_transaction(action)

    def __make_transaction(self, action):
        self.action_history.append(action)
        if action == 'buy':
            self.__buy()
        elif action == 'hold':
            self.__hold()
        elif action == 'sell':
            self.__sell()
        elif action == 'wait':
            self.__wait()
        else:
            sys.exit('Action not implemented, exiting program!')

    def get_investment_performance(self):
        self.return_on_investment = (self.account_history[-1] - self.init_investment) / self.init_investment
        self.profit_on_investment = self.account_history[-1] - self.init_investment
        return {'return': self.return_on_investment,
                'profit': self.profit_on_investment}

    def plot_trading_history(self, stock_prices, date):
        date = date.iloc[-len(stock_prices-1):]
        stock_prices = np.insert(stock_prices, 0, stock_prices[0])
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(40, 20))
        ax1.plot(stock_prices, color='black', label='Cena zamknięcia akcji')
        actions = pd.DataFrame(self.action_history)
        buy_idx = actions[actions[0] == 'buy'].index.to_list()
        sell_idx = actions[actions[0] == 'sell'].index.to_list()
        stock_prices = np.array(stock_prices)
        ax1.scatter(buy_idx, stock_prices[buy_idx], color='green', s=40, label='Kupno')
        ax1.scatter(sell_idx, stock_prices[sell_idx], color='red', s=40, label='Sprzedaż')
        ax1.legend()
        ax2.plot(self.account_history[:-1], label='Kapitał')
        plt.xlabel('Krok czasowy')
        ax1.set_ylabel('Cena akcji')
        ax2.set_ylabel('Kapitał')
        ax2.legend()
        plt.show()

    def __calculate_daily_profit(self):
        self.__actual_investment += self.__actual_investment * self.stock_returns[self.step]

    def __buy(self):
        self.__actual_investment = self.account_history[self.step]
        self.__calculate_daily_profit()
        self.account_history.append(self.__actual_investment)

    def __hold(self):
        self.__calculate_daily_profit()
        self.account_history.append(self.__actual_investment)

    def __sell(self):
        self.account_history.append(self.__actual_investment)
        self.__actual_investment = 0

    def __wait(self):
        self.account_history.append(self.account_history[self.step-1])

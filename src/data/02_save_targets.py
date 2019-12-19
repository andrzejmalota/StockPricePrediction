from src.utils.io import load, save
import pandas as pd


def save_targets():
    data = load('../../data/raw/stock_data.pickle')
    save(pd.DataFrame(data['tesla'][['Date', 'Close']], columns=['Date', 'Close']), '../../data/processed/targets.pickle')


if __name__ == '__main__':
    save_targets()

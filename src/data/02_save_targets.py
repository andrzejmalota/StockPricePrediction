from src.utils.io import load, save


def save_targets():
    data = load('../../data/raw/stock_data.pickle')
    save(data['tesla']['Close'], '../../data/processed/targets.pickle')


if __name__ == '__main__':
    save_targets()

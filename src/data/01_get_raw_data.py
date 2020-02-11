import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from src.utils.io import save


def get_raw_stock_data():
    columns = {'Open': "1. open",
               'High': "2. high",
               'Low': "3. low",
               'Close': "4. close",
               'Volume': "5. volume"}

    api_requests = {
        'tesla': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TSLA&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR',
        'google': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GOOGL&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR',
        'bmw': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=FRA:BMW&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR',
        'daimler': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ETR:DAI&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR',
        'porshe': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ETR:PAH3&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR',
        'amazon': 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&interval=60min&outputsize=full&apikey=CMSZIWYWAHTR01BR'}

    raw_stock_data = {}

    for company, request in api_requests.items():
        req = requests.get(request)
        data = req.json()
        num_rows = len(list(data['Time Series (Daily)'].keys()))
        num_cols = 6
        array = np.full((num_rows, num_cols), fill_value=np.NaN)

        df = pd.DataFrame(data=array, columns=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'])

        for i, (date, values) in enumerate(data['Time Series (Daily)'].items()):
            row = [date, values[columns['Close']], values[columns['Open']], values[columns['High']],
                   values[columns['Low']], values[columns['Volume']]]
            df.iloc[(num_rows - i - 1), :] = np.array(row)

        df[['Close', 'Open', 'High', 'Low', 'Volume']] = df[['Close', 'Open', 'High', 'Low', 'Volume']].apply(
            pd.to_numeric)
        df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
        #     df.index = df['Date']
        #     df.drop('Date', axis=1, inplace=True)
        raw_stock_data[company] = df
    save(raw_stock_data, '../../data/raw/stock_data.pickle')
    return raw_stock_data


if __name__ == '__main__':
    raw_stock_data = get_raw_stock_data()

#
# plt.figure(figsize=(24,8))
# tesla = raw_stock_data['tesla']
# other = raw_stock_data['daimler']
# start = other[other['Date'] == tesla['Date'][0]].index[0]
# print(start)
# tesla_close = tesla['Close']
# other_close = other['Close'][start:].reset_index().drop('index', axis=1)
# plt.plot(tesla_close)
# plt.plot(other_close)
# plt.title("Tesla's daily closing stock price")
# plt.xlabel('Date')
# plt.ylabel('USD')
# plt.legend(['Tesla', 'bmw'])
# plt.show()
#
#
#
# import seaborn as sns
# plt.clf()
# plt.figure(figsize=(24,8))
# sns.distplot(raw_stock_data['tesla']['Close'], bins=40)
# plt.title('Closing price distribution')
# plt.show()

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import check_seasonality

from darts import TimeSeries
import yfinance as yf
# import yfinance_cache as yfc

def retrieval(ticker: str, start_date: str, end_date: str):
    """
    Function fetches data via yahoo finance and transforms it for modeling
    args:
        stock symbol, ticker, string
        stock start date, start_date, string
        stock end data, end_date, string
    returns:
        series, transformed input data
    """
    print('Downloading data...')
    data = yf.download(tickers = ticker, start=start_date, end=end_date)
    print('Done.')
    print(data.head())
    data = data.reset_index()
    try:
        df = data.rename(columns={'Date': 'date', 'Close': 'close'})
    except:
        pass
    import warnings
    warnings.filterwarnings('ignore')

    df['datetime'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype(str).apply(lambda x: x[:10])
    df['time'] = df['date'].astype(str).apply(lambda x: x[11:])
    #
    print(df.head())
    print(df['date'].max(), df['date'].min())
    df['date'] = pd.to_datetime(df['date'])
    df = df[['close','date']]
    series = TimeSeries.from_dataframe(df[['close', 'date']], time_col='date' , freq='D')
    series = df[df.columns[0]]
    return series

def data_prep(data, split=0.8, show_plot=True):
    """
    Function prepares data for machine learning
    """
    tr = round(int(len(data) * split))
    train, test = data[:tr], data[tr:]
    print('full dataset size:', len(data))
    print('80% of the data:', len(data) * split)
    print('training data size:', len(train))
    print('testing data size:', len(test))
    
    if show_plot:
        train.plot(label='train')
        test.plot(label='test')
        plt.show()

    return train, test

def scale_data(train_data):
    # Scale data
    train_scaler = Scaler()
    print('scaling data')
    scaled_train = train_scaler.fit_transform(train_data)
    print('scaled train', scaled_train, 'len:', len(scaled_train))

    return scaled_train
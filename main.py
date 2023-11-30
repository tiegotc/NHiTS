import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from darts.utils.statistics import check_seasonality
# from darts.models.forecasting.baselines import NaiveSeasonal
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from utils import WeekendFillers #fill_weekend_data
import yfinance as yf

data = yf.download(tickers = "AAPL", start="2017-01-01", end="2023-10-30")
print(data.head())
data = data.reset_index()
try:
    df = data.rename(columns={'Date': 'date', 'Close': 'close'})
except:
    pass
# df['datetime'] = df
from darts import TimeSeries
#
import warnings
warnings.filterwarnings('ignore')
#
#df = pd.read_csv('./data/x.csv', sep='\t')
df['datetime'] = pd.to_datetime(df['date'])
df['date'] = df['date'].astype(str).apply(lambda x: x[:10])
df['time'] = df['date'].astype(str).apply(lambda x: x[11:])
#
print(df.head())
print(df['date'].max(), df['date'].min())
df['date'] = pd.to_datetime(df['date'])
df = df[['close','date']]
# df = df[df['date'] >= '2020-01-01']
# filler = WeekendFillers(df)
#
# df = filler.fill_weekend_data()
#df = df.reset_index()
#df = df.rename(columns={'index': 'date'})
#df = df.dropna()
# print('data after fill', df.columns, df.head(15))
#series = TimeSeries.from_dataframe(df[['close', 'date']], time_col='date' , freq='D')
series = df[df.columns[0]]
#series = pd.Series(data=inputs)
print('series:', series)
# series.plot()
# plt.show()
tr = round(int(len(series) * 0.80))
train, test = series[:tr], series[tr:]
print('full:', len(series))
print('80% full:', len(series) * 0.80)
print('train:', len(train))
print('test:', len(test))

train.plot(label='train')
test.plot(label='test')
plt.show()
#
# #############BASELINE MODEL############
# naive_seasonal = NaiveSeasonal(K=len(train))
# naive_seasonal.fit(train)
#
# pred_naive = naive_seasonal.predict(len(test))

# test.plot(label='test')
# pred_naive.plot(label='Baseline')
# #
# #
# naive_mae = mae(test, pred_naive)
#
# #######################################
# #############N-HiTS Model##############
train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)
print('scaled train', scaled_train, 'len:', len(scaled_train))
nhits = NHiTSModel(input_chunk_length=len(train) - len(test), output_chunk_length=len(test), random_state=42)

nhits.fit(scaled_train, epochs=10000)

scaled_pred_nhits = nhits.predict(n=len(test))
pred_nhits = nhits.predict(n=120)
pred_nhits = train_scaler.inverse_transform(scaled_pred_nhits)

mae_nhits = mae(test, pred_nhits)

# print(f'baseline: {naive_mae}, N-Hits: {mae_nhits}')
print(f'N-Hits: {mae_nhits}')
train.plot(label='train')
test.plot(label='Actual')
# pred_naive.plot(label='Baseline')
pred_nhits.plot(label='N-HiTS')

plt.show()


from data import retrieval, data_prep, scale_data
from model import train, predict
from darts import TimeSeries

def run():
    df = retrieval(ticker='MSFT', start_date='2017-01-01', end_date='2020-12-12')
    train, test  = data_prep(data=df, show_plot=False)
    scaled_train = scale_data(train_data=df)
    model = train(scaled_train_data=scaled_train, test_data=test, eps=10)
    predict(model=model, input_data=test)

if __name__ == "__main__":
    run()